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
| M2 | forward 로짓 패리티 (mcore vs Pai 검증 HF 참조) | `tools/verify_forward_parity.py` + `tools/gen_hf_reference_logits.py` | 영/한/887토큰 3종 argmax 일치 + cos ≥ 0.99. 참조 (bf16) 의 최댓값이 정확히 같은 토큰이 2 개면 어느 쪽이든 일치, 3 개 이상이면 퇴화한 참조로 FAIL (2026-11-27, M4 와 같은 규칙 · 허용 폭 없음). top5 일치 · `KL(HF\|\|mcore)` · 로짓 이동은 출력만 한다 (도구는 처음부터 top5 를 판정에 쓰지 않았다) | 2026-08-13 cos ≥ 0.99988 · 2026-08-18 FlashQLA 활성 3종 PASS (TileLang 실행 확인) · 2026-10-06 iter2400 argmax 3/3, cos ≥ 0.99984 · 2026-10-08 iter2862 argmax 3/3, cos ≥ 0.99980 · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` **PASS** (`$NRL_ROOT/gates/codemath2_step282`, MOPD1 학생 재사용 2026-11-10) · 2026-11-12 RL 반출 `mopd1/hf/hf_step_00120` argmax 3/3, cos ≥ 0.99937 **PASS** |
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

### S3 — SWE E2E 학습 경로의 Generation KL, R3 없음 (2026-11-26 측정 · 판정 없음, `$NRL_ROOT/gates/swe_e2e_s3/`)

- **무엇을 재나**: `teacher_swe_e2e_alpha.yaml` 그대로 (Gym `swe_agents` 다중 턴 · 200 턴 · 1,800 s · 131,072 · 세 틀 혼합 · prefix caching 켬 · **R3 끔** · async · 궤적 나이 1) 의 실제 학습 경로에서 롤아웃 엔진과 학습기의 logprob 차이. R3 는 지금 핀에서 `swe_agents` 와 함께 쓸 수 없다 (`E2E_SANDBOX.md` §7.9). 사용자 지시 2026-11-26 "R3 없이 SWE E2E 로 KL 측정을 바로 진행해".
- **기준을 적용하지 않는다.** 값을 보고 사용자가 정한다 (R1 의 기준은 0.001, 결정 80).
- **명령**: `EXTRA_OV="loss_fn.seq_logprob_error_in_loss=false" $NRL_ROOT/gates/swe_e2e_s3/run_s3b.sh s3b <teacher_blends/teacher_swe_e2e_smoke12.jsonl> 4 8 3` — 래퍼는 SWE 기동 전 점검 → `runs/launch.sh` → 컨테이너 정리 순서다. W&B 끔 · 체크포인트 저장 끔 · 시작점 `mopd1/hf/hf_step_00120` · commit `00393359d` · 68 분 · rc 0.
- **분석 (CPU)**: `analyze_s3.py <run dir> --steps 3 --prompts 4 --gens 8` (쓴 사본 `s3b/analyze_s3.py.used`, 결과 `s3b/analysis.txt` · `analysis.json`). 덤프의 토큰별 logprob 으로 KL 을 다시 계산해 TensorBoard 값 · 분모와 대조하고, 빠진 행의 사유를 하니스 기록으로 붙인다.

| step | KL (덤프 = TensorBoard, 차이 0.001%) | 유효 시퀀스 | 유효 토큰 | 빠진 행 | 궤적 나이 | lr | 평균 보상 |
|---|---|---|---|---|---|---|---|
| 1 (on-policy) | **0.001555** | 27/32 | 409,903 | 5 (턴 상한 4 · 문맥 상한 1) | 0 | 1.0e-7 | 0.31 |
| 2 | 0.001637 | 25/32 | 312,592 | 7 (문맥 상한 4 · 턴 상한 3) | 1 | 3.9e-7 | 0.34 |
| 3 | 0.001658 | 29/32 | 403,294 | 3 (턴 상한 3) | 1 | 6.8e-7 | 0.41 |
| 합산 | **0.001615** · 95% 구간 [0.001513, 0.001721] (프롬프트 그룹 12 개 bootstrap) · 표준오차 0.000055 (과제 jackknife) | 81/96 | 1,125,789 | 15 | — | — | — |

| 분해 (합산) | KL |
|---|---|
| 과제별 (12) | 0.00127 ~ 0.00211, 12 개 모두 0.001 초과 |
| 시퀀스별 (81) | 최소 0.00088 · 중앙 0.00173 · 최대 0.00264, mean exp\|d\| 최대 1.040 · logprob 필터 (> 2) 0 건 |
| 문맥 위치 | 0 ~ 16K 0.00203 · 16 ~ 32K 0.00174 · 32 ~ 64K 0.00145 · 64 ~ 98K 0.00119 · 98K ~ 0.00166 (4 시퀀스) |
| 턴 순번 | 1 ~ 5 0.00231 · 6 ~ 20 0.00189 · 21 ~ 50 0.00159 · 51 ~ 100 0.00141 · 101 ~ 0.00139 |
| 틀 | CodeAct 0.00166 (28) · Codex 0.00156 (24) · OpenCode 0.00166 (25) |

- **읽는 법**: on-policy 값은 step 1 의 0.00156 이다. 합산 0.00162 는 lr 워밍업 구간 (1e-7 ~ 7e-7, 운영 3e-6) 이라 가중치가 거의 같은 세 step 을 합친 값이다. step 사이의 차이 (5 ~ 7%) 는 step 마다 과제가 달라서 생기는 퍼짐과 구분되지 않는다 — 상승 경향으로 읽지 않는다. 단일 턴에서 R3 없이 잰 값 (0.0015, "R1 2×2") 과 같은 수준이고 R3 를 켠 값 (0.0006) 의 약 2.7 배다.
- **문맥을 따라 커지지 않는다**: 같은 시퀀스 안에서 뒤쪽 (32K 이상) 과 앞쪽 (16K 미만) 의 KL 비율은 0.67 [0.60, 0.75], 토큰 구성 (surprisal) 을 맞춘 뒤 0.83 [0.74, 0.92] 이고 55 시퀀스 중 46 개가 뒤쪽이 낮다. 재귀 상태 · 캐시 복원으로 오차가 쌓이는 징후는 없다 (같은 가중치에서 캐시를 복원하는 경우에 한한 말이다). 위치별 표의 하락 가운데 절반쯤은 토큰 구성의 효과다.
- **이 값이 말하지 않는 것 (미측정)**: ① 운영 lr 에서 가중치 갱신 뒤 낡은 캐시의 영향 — 갱신 때 캐시를 비우지 않는다 (`Successfully reset prefix cache` 는 기동 직후 첫 refit 에만 찍힌다). step 2 의 궤적은 갱신을 넘지 않았고 step 3 만 넘었으며 그 비율은 재지 않았다 ② prefix cache 적중률 — 학습 경로에 수집 수단이 없다. 간접 확인만: 응답 64 토큰 이하 호출의 시간 중앙값이 프롬프트 16K 미만 0.42 s → 96K 이상 1.12 s (캐시를 끈 fleet 은 64K 이상 3.7 s) ③ 상한에 걸려 마스킹된 롤아웃 (15/96, 턴 상한 10 · 문맥 상한 5) ④ SWE 에서 R3 를 켠 대조군.
- **경로 확인 (s3a, 과제 2 × 2 · 1 step, commit `a8a392bbd`)**: 처음 도는 경로 (`swe_agents` ↔ 학습 엔진의 토큰 ID 전달 · 결과 조립 · 학습 step · refit) 통과, KL 0.001836 (4 시퀀스 · 75,283 토큰). 두 런 모두 토큰 연속성 · 빈 롤아웃 예외 0 · 배치 재시도 0 · 남은 컨테이너 0.
- **독립 검토**: 측정 설계 — "보완 필요" (step 2 ~ 3 은 워밍업 lr 이라 낡은 캐시 효과를 못 잰다 · 적중률 수집 수단 없음 · KL 분모가 로그에 없다 · 표본 16 은 빠듯) → 반영 → 실행 설정 "진행 가능". 분석 도구 — 네 번 고친 뒤 "신뢰 가능": KL 을 계산하지 못한 step · 빠진 step · 유효 시퀀스 0 · 사유 없는 마스킹 · 상한에 걸렸는데 KL 에 남은 행이 모두 "문제 없음" 으로 통과하던 경로를 막았고 음성 대조로 확인했다. 재표본 단위는 시퀀스가 아니라 프롬프트 그룹이다 (한 프롬프트의 8 개는 같은 과제). 보고 문장 대조 — "on-policy" 는 step 1 에만 쓴다 · 위치별 하락의 절반은 구성 효과 · 갱신을 넘은 것은 step 3 뿐.
- **다시 잴 때**: R3 전달 경로가 생기면 같은 명령으로 대조한다. `seq_logprob_error_in_loss=false` 에서는 롤아웃이 전부 마스킹돼도 런이 죽지 않고 KL 0 이 기록된다 — 유효 시퀀스 수 (`train/global_valid_seqs`) 를 함께 읽는다.

### S4 — SWE E2E 본 런의 첫 step 판정 (2026-11-26 정의, `RL_PLAN.md` 결정 85 · 86 · 90 · 93)

별도 런 없이 본 런 (`teacher_swe_e2e` / `s4`, 8 × 8 · 875 step) 의 첫 step 들로 판정하고, 통과하면 그대로 학습을 잇는다. step 덤프는 쓰지 않는다 — TensorBoard · 드라이버 로그 · 그룹 보상 로그 (`rollout_groups.jsonl`) · 하니스 결과로 본다. 실행 `$NRL_ROOT/gates/swe_e2e_s4/run_s4.sh` · 정지 `stop_s4.sh` (기본은 다음 체크포인트가 완성된 뒤).

| 단계 | 시점 | 기준 | 어긋나면 |
|---|---|---|---|
| G1 | step 1 ~ 2 | `check_g1.py <run dir> --steps 2` 가 0: 예외 0 · `gen_kl_error` < 0.002 · `num_masked_seqs_by_logprob_error` 0 · 유효 시퀀스 32 이상 · step 1 의 lr 1e-7 · `weight_sync` > 0 과 "Updated weight version to s" · 하니스 거부 지표 > 0 · 그룹 8 개 × 롤아웃 8 개가 하니스 기록과 전부 짝지어지고 (생성 토큰 수) 과제 하나 · 첫 프롬프트 길이 하나 · 짝지은 롤아웃의 거부 사건 합 = `num_harness_rejected_tool_calls/mean` × 64 · R3 꺼짐 · prefix caching 켬 · 틀 고정 · 거부 문구 정규식 | 중단 · 진단 · 보고 |
| G2 | step 2 ~ 10 | 기준선과 step 간 잡음을 잰다: `gen_kl_error / approx_entropy` · `(token_mult_prob_error − 1) / approx_entropy` · `approx_entropy` · `gen_tokens_per_sample/mean` · `max_gen_tokens_per_turn/mean` · `turns_per_sample/mean` · 롤아웃당 하니스 거부 수 · 마스킹 비율 (1 − 유효 시퀀스 / 64) · 해결률 · step 주기 (짧은 step 과 긴 step 따로) | 예외면 중단 |
| G3 낡은 캐시 · off-policy | step 11 ~ 30 | 두 비율의 10 step 이동평균이 기준선의 +20% 를 넘으면 보고 (이 비율은 낡은 캐시 없이도 20% 넘게 움직인 적이 있어 정지 기준이 아니다). 어느 step 이든 `gen_kl_error` > 0.004 면 중단. `num_masked_seqs_by_logprob_error` 가 0 이 아니면 보고 | 보고 (대응 후보: 갱신 때 KV 재계산 · lr 낮춤) |
| G4 벌점 부작용 | step 11 부터 10 step 마다 | `approx_entropy` · `gen_tokens_per_sample/mean` · `max_gen_tokens_per_turn/mean` · `turns_per_sample/mean` 의 10 step 이동평균이 기준선 ±30% 를 벗어나거나, 거부 수가 줄지 않은 채 해결률이 떨어지면 보고하고 −2 로 내릴지 묻는다 | 곡선과 함께 보고 |
| G5 상태 | 상시 | 마스킹 비율 40% 초과 5 step 연속 · step 주기가 G2 에서 잰 긴 step p95 의 1.5 배 초과 (저장 step 제외) · 그 2 배 동안 step 없음 = 정지 (엔진별 생성 토큰 증가량으로 낙오와 교착을 가른다) · 호스트 메모리 (`ray/node.*.mem_gb`) · NFS · 홈 여유 · 반출 실패 (`hf_s4/*.failed`) | 진단 · 보고 |

- **잡음 바닥**: step 당 KL 1σ 약 5% (과제 8 개) · 엔트로피 CV 2.5 ~ 5% · 생성 토큰 6 ~ 12% · 턴당 최대 16 ~ 40% (앞선 실제 런). KL 은 엔트로피를 따라 움직인다 (상관 0.83 ~ 0.98) — 그래서 비율로 본다. `probs_ratio*` (`force_on_policy_ratio` 라 늘 1) 와 `avg_trajectory_age` (늘 1) 는 쓰지 않는다.
- **이 판정이 보지 못하는 것**: 엔진이 갱신된 가중치로 정확히 계산하는지 (G3 이 유일한 안전장치) · prefix cache 적중률 · 상한에 걸려 마스킹된 롤아웃 · 롤아웃의 마지막 호출이 거부된 경우.
- **독립 검토**: 계획 — 1 차 "보완 필요" (KL 임계 0.0025 가 잡음과 동떨어지고 엔트로피에 가려진다 · 벌점 관문이 한 방향만 본다 · 체크포인트 값이 레시피에 없다) → 반영. 준비물 — "보완 필요" (하니스 결과 위치 · 최근 10 개만 남는 보존 · 재개 시 데이터 부족 · 점검기 오경보) → 반영 → "반출 감시 수정 둘이면 신뢰 가능" → 반영. `check_g1.py` 는 스모크 `s3c` 에서 통과하고 수정 전 `s3b` 에서 문제 18 건을 낸다.
- **G1 결과 (2026-11-26 17:55, step 1 · 2)**: 지표 기준은 전부 통과했다 — KL 0.00172 / 0.00169 · logprob 필터 0 · 유효 38 / 45 · step 1 의 lr 1e-7 · 가중치 갱신 · 예외 0 · 그룹 8 × 8 · 과제마다 첫 프롬프트 길이 하나. 대조 두 항목은 **불일치** 였다 — 짝지은 롤아웃 60/64 · 59/64, 하니스 거부 수 지표 111 대 이력 103 · 66 대 56 (`$NRL_ROOT/gates/swe_e2e_s4/s4/g1.txt`).
  진단 (`problem-diagnosis`): 점검기의 빈틈이다. 하니스가 거부된 턴을 버린 것도, 학습기 표지가 틀린 것도 아니다. ① 하니스 이력이 없는 롤아웃 9 개 — 에이전트 시간 초과 5 (`output.jsonl` 없음) · 패치 추출 실패 4 (`Failed to cd to /<repo>: … previous command is still running` 뒤 이력이 빈 출력). 학습기는 궤적을 호출 로그에서 받으므로 (`swe_agents/app.py:3863`) 이들도 센다. 점검기는 사용량도 이력도 없는 기록을 건너뛰었다. ② 문맥 한도로 끝난 롤아웃의 마지막 호출 거부 4 건 — 하니스가 오류를 붙이고 한 번 더 호출을 시도해 (0 토큰) 오류 응답이 마지막 요청에 들어 있다. 점검기는 "마지막 호출의 거부" 를 일괄로 뺐다. 산술: 103 + 6 + 2 = 111 · 56 + 8 + 2 = 66.
  선행 사례: upstream Gym 은 `max_iteration` · `context_window` · 시간 초과 · OOM 만 마스킹한다 (`app.py:3852-3859`). 프론티어 보고서 · Ultra 레시피에는 해당 없음 (점검기 문제).
  독립 검증 (증상만 받은 새 에이전트 → 진단서 반박, 두 단계): **확인**. 학습기 조립을 호출 로그에서 재현해 step 1 · 2 의 64/64 롤아웃이 `gen_tokens` 와 맞고 거부 합 111 · 66 이 재현됐다. 턴 누락 배제 (이력 있는 119 개 전부 하니스 토큰 합 = 학습기 토큰 합, 학습기 표지 수 = 이력의 거부 사건 105/105 · 58/58), 표지 오류 배제 (표지 없는 거부 문구 0 · 이름 불일치 0). 반론: 합계 논리 ("지표가 더 크다") 로는 배제가 안 되고 롤아웃 단위의 두 사실이 근거다.
  보완 후보 `check_g1_v2.py` 에 대한 판정은 "고칠 것이 있다": 같은 과제 안의 소거 짝짓기는 개수만 맞춰 턴 누락을 가릴 수 있다 (호출 로그로 조립을 재현해 토큰 수가 같은지 요구해야 한다) · "끝남" 판정이 지표 파일 기록과 경합한다 · 이력이 빈 까닭을 문구로 확인하지 않는다 · 독립으로 확인되는 몫 (105/111 · 58/66) 에 하한이 없다. **점검기 변경은 승인 대기** — 그 전까지 G1 의 공식 판정은 "지표 기준 통과 · 대조는 독립 검증자의 재현으로 확인" 이다.
  런은 멈추지 않았다 (계획은 "어긋나면 중단" — 지표 기준이 통과했고 불일치 방향이 턴 누락과 반대여서 진단부터 했다).
- **본 런에서 새로 보인 것 (step 1 ~ 4, 2026-11-26)**: ① 해결률 7.8% · 3.1% · 0% · 6.3% (과제 32 개 평균 4.3%), 보상 분산이 있는 그룹 1/8 · 1/8 · 0/8 · 1/8. ② 벌점 비중 (토큰별 advantage 절댓값 합에서 −5 의 몫, 그룹 로그와 하니스 기록으로 추정 — 유효 토큰 수가 step 1 · 3 에서 TensorBoard 와 정확히 같다): 50.1% · 15.1% · 100% · 24.1%, 합계 37.8% (스모크 `s3c` 는 22.4%). −2 면 약 20%, −1 이면 약 11%. ③ 패치 추출 실패 롤아웃은 마스킹되지 않고 보상 0 으로 학습된다 (step 1 · 2 에 4/128, 끝난 기록 272 개 중 7 개).
- **감시 스크립트** (`$NRL_ROOT/gates/swe_e2e_s4/monitor_s4.py <run dir>`, 종료 코드 0 = 보고할 것 없음 · 2 = 보고 · 3 = 중단 조건): 이동평균 창은 기준선 뒤 최근 10 step 이고 5 step 이상 찼을 때만 판정한다 (G3 · G4 의 첫 판정은 step 15 — 그 전에는 KL > 0.004 중단 조건이 지킨다). 부를 때마다 기준선 뒤의 모든 창을 다시 본다. 값이 NaN · inf 면 중단 조건이다. "batch did not complete prompt groups" (재시도에 성공해도 찍힌다) 와 그룹 로그 기록 실패는 보고로 둔다. 재개하면 새 런의 첫 step 이상인 옛 기록은 버리고, 재개 첫 step 은 주기 판정에서 뺀다. 저장 step (20 의 배수) 은 주기 판정에서 뺀다 — 저장은 그 step 의 기록 전에 일어난다.
  독립 검토: 1 차 "보완 필요" (NaN 이 비교를 통과 · 재개 때 낡은 step 이 섞임 · 마지막 10 step 만 봐서 호출 간격이 벌어지면 이탈을 놓침) → 반영. 2 차 "두 곳만 고치면 신뢰 가능" (자체 시험의 재개 검사 식이 아무것도 확인하지 않음 · 이벤트가 없는 새 exp 의 재개 시각) → 반영. 자체 시험 (`--selftest`): 잡음 없는 곡선 27 가지에서 기대한 문구만 나온다.
- **G4 의 두 규칙은 잡음 안에 있다 (2026-11-26, 앞선 런 크기의 step 간 CV 로 합성한 곡선 300 개씩 — 엔트로피 4% · KL 비율 5% · 길이 10% · 턴당 최대 28% · 턴 8% · 거부 15% · 해결률 30%)**. 나머지 규칙 (KL 비율 · mult_prob 비율 · 엔트로피 · 길이 · 턴 수) 은 정상 곡선 오경보 0% · 결함 검출 100% 다.

  | 규칙 | 정상 곡선 오경보 (30 / 110 / 300 step) — 턴당 최대 | — 해결률 | 검출 (30 step): 턴당 최대 +45% · +100% | 해결률 −40% · −60% |
  |---|---|---|---|---|
  | 쓰는 규칙 (±30% · 2 SE) | 10% / 30% / 46% | 12% / 30% / 42% | 94% · 100% | 97% · 100% |
  | 후보: \|z\| ≥ 3.5 이고 변화 ≥ 15% · 해결률 3 SE 이고 상대 하락 ≥ 25% | 5% / 12% / 17% | 2% / 7% / 13% | 77% · 100% | 82% · 100% |
  | 후보: 쓰는 규칙이 5 창 연속 | 3% / 11% / 15% | 3% / 10% / 20% | 87% · 100% | 92% · 99% |

  원인은 고정 띠가 지표의 잡음 크기와 맞지 않는 것이다 (턴당 최대는 창 평균과 기준선 평균의 잡음을 합쳐 1σ 약 13% 라 ±30% 는 약 2.3σ, 겹치는 창이 많다). 기준선이 9 step 이라 그 표준편차도 ±25% 쯤 틀릴 수 있다. 둘 다 중단이 아니라 보고 조건이다. **기준은 바꾸지 않았다** — step 10 에 이 런의 CV 로 다시 재서 (`--selftest --noise-from <run dir>`) 선택지와 함께 보고한다.

### S5 — SWE E2E 재시작 `teacher_swe_e2e/s5` (32 × 8, `s4` step 60 가중치에서, 2026-11-27 · `RL_PLAN.md` 결정 94)

- **시작 체크포인트 게이트** (재실행 조건 "RL 반출 체크포인트를 시작점으로" — M1 · M2 · M4 → R1, M3 는 기록): `$NRL_ROOT/gates/s4_step60/run_gates.sh` (`mopd1_step120` 과 같은 명령, R1 기준은 결정 80 의 0.001). **결과 (2026-11-27 10:44 ~ 11:20, `hf_s4/hf_step_00060`, commit `0036e35d4`): GATES PASS** — M1 전 가중치 일치 · M2 argmax 3/3 (top5 2/3), cos ≥ 0.999355 · M4 3/3 PASS, cos ≥ 0.999987, KL(HF‖vLLM) ≤ 0.0085 · M3 (기록) 평균 |d| 0.0565 · R1 KL 0.0006 / 0.0006 / 0.0006. HF 반출 대조: 14,181 텐서 · 동결 텐서 변화 0.
- **본 런의 첫 step 판정** 은 S4 의 정의를 그대로 쓰되 step 수를 과제 수에 맞춰 옮긴다 (step 당 과제 8 → 32). 기준 값 자체는 같다.

| 단계 | S4 (8 × 8) | S5 (32 × 8) |
|---|---|---|
| G1 | step 1 ~ 2, 유효 시퀀스 32 이상 (64 중) | step 1 ~ 2, 유효 시퀀스 128 이상 (256 중). 점검기는 모든 롤아웃을 학습기 조립 재현으로 짝짓는다 (결정 94 ⑤) — 이력 없는 롤아웃이 step 의 12.5% (32 개) 를 넘으면 문제 |
| G2 기준선 | step 2 ~ 10 (과제 72) | step 1 ~ 3 (과제 96, 워밍업 구간) |
| G3 · G4 창 | 기준선 뒤 최근 10 step, 5 step 이상일 때 판정 | 기준선 뒤 최근 3 step (과제 96), 2 step 이상일 때 판정 — 첫 판정 step 5 |
| G5 마스킹 | 40% 초과 5 step 연속 (롤아웃 320) | 40% 초과 2 step 연속 (롤아웃 512) |
| G5 주기 · 정지 | 긴 step p95 × 1.5 · × 2 | 같다 (모든 step 이 긴 step, 약 68 분). 저장 step 은 5 의 배수 |
| 어느 step 이든 | KL > 0.004 · NaN · 예외 → 중단 | 같다 |

- 감시 스크립트 `monitor_s4.py` 에 창 크기 인자를 더해 쓴다 (`--base 1 3 --win 3 --min-win 2 --mask-run 2 --save-period 5 --prompts 32`). 창 재조정과 점검기는 **판정에 쓰기 전에 적대적 검토** 를 받는다 (step 2 는 기동 약 2.5 시간 뒤, 첫 G3 · G4 판정은 약 6 시간 뒤).
- **점검기 `$NRL_ROOT/gates/swe_e2e_s5/check_g1.py`** (결정 94 ⑤ — 독립 검증자의 방식): 롤아웃마다 학습기의 궤적 조립을 호출 로그에서 재현해 생성 토큰 합으로 짝짓고, 재현한 표지 수의 합이 TensorBoard 지표 × 롤아웃 수와 같은지 본 뒤, 하니스 이력이 있는 롤아웃은 **호출 (tool_call_id) 단위** 로 대조한다 — 버려진 턴 (이력의 거부 호출이 궤적에 없다) · 표지 누락 · 거짓 표지가 모두 0 이어야 한다. 이력이 없는 롤아웃은 step 의 12.5% 이하이고 까닭이 알려진 것 (에이전트 시간 초과 · 이력이 빈 출력 + 알려진 오류 문구) 이어야 한다.
  시험 (2026-11-27): `s4` step 1 · 2 — 짝지음 64/64 · 64/64, 표지 111 = 지표 111.0 · 66 = 66.0, 이력 있는 60 · 59 개에서 거부 호출 105/105 · 58/58 확인, 버려진 턴 · 표지 누락 · 거짓 표지 0, 이력 없는 4 (6.2%) · 5 (7.8%). 128 롤아웃에 5.5 분 (256 × 2 면 약 20 분, `nice` 로 돌린다). 드라이버 로그 전체를 세는 `Traceback` 줄만 걸렸다 (뒤 step 의 하니스 쪽 2 건 + 정지 때 1 건).
  적대적 검토: **신뢰 가능**. 턴 누락 · 토큰 ID 누락 · 잘못된 파일 선택 · 마지막 응답 중복은 모두 소리 나게 실패한다. 한계 둘 — ① 학습기가 *어느 턴* 에 −5 를 걸었는지는 개수로만 본다 (턴 위치는 덤프가 있는 `s3c` 의 61/61 로만 확인) ② 같은 과제의 재시도분이 같은 토큰 합이면 학습에 쓰이지 않은 디렉토리를 고를 수 있다 (가능성 낮음).
- **감시 `$NRL_ROOT/gates/swe_e2e_s5/monitor.py`** (`monitor_s4.py` + 창 크기 인자). 적대적 검토: 창 재조정은 타당 (S5 의 step 하나 = S4 의 step 넷, 2 step 창 = 과제 64 · 3 step 기준선 = 과제 96 · "40% 초과 2 step 연속" = 롤아웃 512), **해결률 2 SE 규칙은 보완 필요** — 기준선 3 step · 창 2 step 이면 step 단위 표준편차의 자유도가 1 ~ 2 라 판정이 무작위가 된다 → 그룹 로그의 **과제 단위** 평균 보상으로 SE 를 내도록 고쳤다 (분산은 기준선 과제의 것을 양쪽에 쓴다 — 창이 전부 0 이면 분산이 0 으로 나와 거짓으로 걸린다). 자체 시험: 30% → 30% 조용 · 30% → 5% 검출 · 3% → 0% 는 바닥이라 검출되지 않음. 남는 것: 주기 기준 p95 는 step 2 · 3 의 값 둘뿐이라 거칠다 · 다른 지표의 σ 주석은 기준선 3 step 이라 거칠다 · 창이 약 215 개라 오경보가 는다 (검토 권고: |z| ≥ 3.5 이면서 변화 15 ~ 20% 이상 — 기준 변경이라 적용하지 않았다).
- `s4` 에서 드러난 감시의 한계를 그대로 안고 간다: 드라이버 로그의 `Traceback` 문자열은 하니스 쪽 비치명 오류에도 걸린다 (s4 에서 2 건) — 걸리면 내용을 보고 판정한다. G4 의 두 규칙 (턴당 최대 · 해결률) 은 잡음 안에 있다.

### S6 — 미완 궤적 벌점 (`RL_PLAN.md` 결정 95) 의 검증과 본 런 `teacher_swe_e2e/s6` (2026-11-27)

빠른 것부터. 도구는 `$NRL_ROOT/gates/swe_e2e_s6/`.

| 단계 | 무엇을 | 잡는 결함 | 시간 |
|---|---|---|---|
| ① CPU — 가짜 모델 | `run_stub_unfinished.sh <tag> on\|off` + `summarize_stub_unfinished.py`: 실제 컨테이너 · 실제 하니스, 모델만 가짜 (`stub_model4.py` — 홀수 과제는 끝내지 않아 턴 한도 8 에 걸린다). on: 그 롤아웃이 보상 −1 · 마스킹 안 됨 · `unfinished_penalized` 참, 끝까지 간 롤아웃은 보상 1. off (음성 대조): 보상 0 · 마스킹 | 패치의 분기 · 설정 키 전달 · 결과 필드 | 각 5 분 |
| ② GPU 스모크 | `run_smoke.sh` 4 × 8 · 1 step · 스모크 12 과제 · 시작 `hf_s4/hf_step_00040` · step 덤프 켬 → `check_smoke_unfinished.py` | 학습기가 실제로 쓴 값: 벌점 대상의 보상 −1 · 손실 포함 · advantage 재계산 일치 · 지표 일치 · KL. 벌점 대상이 하나도 없으면 통과가 아니다 | 약 40 분 |
| ③ 본 런의 첫 step | `check_g1.py --prompts 32` (S5 의 점검기 + 7 번 검사: 롤아웃마다 벌점 대상 여부 · 그룹 로그의 보상 · 지표 · 마스킹 수) | 문맥 한도 사례 · 256 롤아웃 규모 · 덤프 없는 경로 | step 2 뒤 약 20 분 |
| ④ 본 런 감시 | `monitor.py --prompts 32 --base 1 3 --win 3 --min-win 2 --mask-run 2 --save-period 5 --baseline-from <s4> 2 10`: G3 · G4 의 기준선을 원래 시작 모델의 `s4` step 2 ~ 10 에서 잰다 (이 런 자체의 첫 step 을 기준으로 잡으면 시작 가중치에 든 변화가 가려진다 — `s5`). 양방향 ±30% 목록에 패치 생성률을 더했다 (결정 95 ⑤) | 전환 구간의 조기 종료 (턴 수 · 패치 생성률 하락) · 길이 · KL | 체크포인트마다 |

- 스모크 점검기의 음성 대조 (2026-11-27): 벌점이 없던 스모크 `s3c` 에서 문제 29 건 (벌점 대상 9 개가 보상 0 · 마스킹 · 지표 없음). 같은 실행에서 32 행을 전부 짝지었고 보상으로 다시 계산한 advantage 가 덤프와 32/32 일치했다.
- **적대적 검토 (2026-11-27, 패치 · 스모크 기준 · 본 런 도구)**: 패치의 분기 · 설정 전달 · 기본값은 맞다. 반드시 고칠 것 셋 → 반영.
  ① 학습기 쪽 보상 규칙이 위반 시 보상을 0 으로 덮어써 −1 이 0 으로 *올라간다* (퇴화한 응답에 보상) → 본체 수정 `349a516a6` (`min(보상, 0)`, 단위 테스트 4 개 추가 · 61 통과).
  ② 잘린 시퀀스 필터가 문맥 상한에서 끝난 롤아웃 (문맥 한도 701 개 중 269 개 + 턴당 상한 25 개) 의 손실 가중치를 0 으로 만들어 벌점을 지운다 → 레시피 `overlong_filtering: false`. 스모크는 "문맥 상한에서 끝난 벌점 대상이 1 개 이상이고 손실에 남아 있다" 를 요구하고 (스모크 데이터 `teacher_swe_e2e_smoke_ctx4.jsonl` — `s4` · `s5` 에서 문맥 한도 · 턴 한도에 자주 걸린 4 과제), 본 런 점검은 "유효 시퀀스 = 전체 − 마스킹 − logprob 필터" 와 드라이버 로그의 `overlong_filtering=False` 를 본다.
  ③ 감시의 고정 기준선이 거부 수까지 옛 값으로 바꿔 해결률 규칙이 꺼졌다 → 거부 수는 이 런의 값을 쓴다. 다른 런의 표준편차로 낸 σ 주석은 뺐고, 참고 구간 (`--also-show`, 시작 가중치 무렵의 `s4` step 31 ~ 40) 을 함께 적는다.
  남는 한계: 벌점은 "아무것도 고치지 않고 일찍 끝내기" (0 > −1) 를 유인할 수 있다 — 빈 패치는 패치 생성률이 잡고, 사소하게 고치고 끝내는 쪽은 해결률 규칙만 잡는다 (통과율 2 ~ 3% 에서는 그 규칙이 걸리지 않는다). 적은 턴에서 끝난 실패의 비율 · 끝내기 전 테스트 실행 여부는 감시에 없다.
- 시작 체크포인트 게이트 (`hf_s4/hf_step_00040`, M1 · M2 · M4 → R1): `$NRL_ROOT/gates/s4_step40/`. **결과 (2026-11-27 16:39 ~ 17:11, commit `d7a87798a`): 사슬 FAIL — M2.**

  | 게이트 | 결과 |
  |---|---|
  | M1 | `ALL WEIGHTS MATCHED EXACTLY` |
  | M2 | **FAIL** — 프롬프트 0 · 1 PASS (cos 0.999957 · 0.997915), 프롬프트 2 가 `next=115359 (HF 362) argmax_match=False cos=0.999971 max_diff=0.6250 mean_diff=0.0925` |
  | M4 | 3/3 PASS (프롬프트 2 는 `argmax=115359 (HF 362) match=True hf_max_ties=[362, 115359]`, KL 0.0038 · 0.0005 · 0.0083) |
  | M3 (기록) | mean \|d\| 0.0353 |
  | R1 | KL 0.0006 × 3 (기준 0.001) |

  사슬이 FAIL 이라 가짜 모델 시험 · 스모크 · 기동은 시작하지 않았다.
  - **진단 (독립 검증 "확인")**: HF 참조의 프롬프트 2 최댓값이 두 토큰 (362 ' the' · 115359 " Einstein's") 에서 정확히 같다 (41.75 — bf16 의 32 ~ 64 구간 간격은 0.25). `verify_forward_parity.py` 는 `argmax()` 가 돌려주는 첫 인덱스 하나만 정답으로 봐서, Megatron 이 동률의 다른 쪽을 고르면 FAIL 한다. M4 도구는 2026-08-24 (`83976fe0d`) 부터 같은 프롬프트 · 같은 두 토큰의 동률을 "동률 집합의 어느 원소든 일치" 로 처리하는데 M2 도구에는 옮기지 않았다. 이 프롬프트의 두 토큰은 저장된 참조 다섯 개 모두에서 bf16 한 간격 안이다 (+0.25 · +0.25 · −0.25 · −0.25 · 0.00). 서사는 `KNOWN_ISSUES.md` 2026-11-27 "M2 게이트가 참조의 최댓값 동률에서 FAIL".
  - **검증자가 따로 짚은 것**: ① "Megatron forward 가 정상" 은 증명된 것이 아니다 — M2 의 해상도에서 결함이 보이지 않는다는 것까지다. M4 · R1 통과는 이 프롬프트의 Megatron 대 HF 를 묶어 주지 못한다 (R1 은 라우팅 재생을 켠 채 vLLM 과 견준다). ② 통과한 프롬프트 1 의 cos 0.997915 · mean 0.8226 은 기록상 가장 크다 (이전 0.99935 ~ 0.99980 · 0.16 ~ 0.41). 로짓 전체의 균일한 이동 (KL ≈ 0.0003) 으로도, 최상위 토큰에 몰린 오차 (KL ≈ 0.05) 로도 설명된다 — Megatron 로짓을 저장하지 않아 가를 수 없다. ③ 프롬프트 0 · 2 에서는 `cos ≥ 0.99` 가 거의 아무것도 걸러내지 못한다 (상수 벡터와의 cos 가 0.994) — M2 는 사실상 argmax 게이트다. ④ 이 표의 M2 기준 "top5 일치" 는 도구가 판정에 쓰지 않는다.
  - **M2 의 남는 한계 (고치지 않고 적어 둔다)**: 프롬프트 2 의 두 토큰은 모든 체크포인트에서 bf16 한 간격 안이라, 동률이 아닐 때도 정상 forward 가 뒤집힐 수 있다 (검증자의 잡음 모형으로 한 간격일 때 5 ~ 24%, 실제로는 동률이 아닌 4 건 중 0 건). 로그의 후보 토큰 로짓이 양쪽 모두 한 간격 안이면 그 불일치는 결함의 증거가 아니다. M2 는 이웃 체크포인트 수준의 가중치 차이를 가르지 못한다 (step 60 의 로짓을 step 40 참조로 판정해도 3/3 통과) — 가중치 일치는 M1 이 지킨다.
  - **수정 (사용자 승인 2026-11-27 17:47 "1. 권고대로 진행해" · 적대적 검토 2 회 "보완 필요" → 보완 반영)**: 도구에 동률 규칙 (참조 최댓값이 정확히 같은 토큰 2 개까지, 3 개 이상이면 퇴화한 참조로 FAIL · 허용 폭 없음) · `KL(HF||mcore)` 와 로짓 이동 출력 · `--dump` (Megatron 로짓 저장). 재실행은 M2 만 (`rerun_m2_tierule.sh` — 같은 참조, 새 파일, CPU 대조 둘이 통과해야 GPU 를 잡는다, step 60 을 대조로 함께 잰다, 대조가 실패하면 무효). 기동 조건: 프롬프트별 KL(HF‖Megatron) ≤ 이전에 통과한 사슬 4 건의 M4 KL 최댓값의 2 배 (0.01145 · 0.00087 · 0.01699) — **빌린 기준**이고 Megatron 에 대해 보정되지 않았다 (`m2_dump_check.py` 머리말에 차단을 읽는 법까지 실행 전에 적어 둠). CPU 대조: `m2_rule_check.py` (양성 3 · 음성 6 · 퇴화한 참조 3, 상한을 3 으로 늘린 변형을 잡는다) · `m2_dump_check.py --selftest`. 마지막 검토 판정 "신뢰 가능" (대조가 실패하면 무효 · 차단을 읽는 법을 실행 전에 고정 · 기준은 이 체크포인트 이전에 통과한 사슬 4 건에서).
  - **M2 재실행 결과 (2026-11-27 17:48 ~ 17:52, commit `3432cbb80`, `s4_step40/gates_tierule.log`) — M2 는 통과, KL 기동 조건은 차단.**

    | 항목 | step 40 (시작점) | step 60 (대조) |
    |---|---|---|
    | M2 (동률 규칙) | PASS — 프롬프트 2 `hf_max_ties=[362, 115359] logits(hf/mcore)=362:41.75/41.50,115359:41.75/41.75` | PASS (동률 없음) |
    | KL(HF‖Megatron) 프롬프트 0 · 1 · 2 | 0.002779 · **0.001440** · 0.007854 | 0.001462 · 0.000299 · 0.000624 |
    | 기동 조건 (0.01145 · 0.00087 · 0.01699) | **차단 — 프롬프트 1** (기준의 1.65 배 · 대조의 4.81 배 → 실행 전에 정한 읽는 법으로 "이상") | 통과 |
    | 이전 실행과의 재현 (next · cos · max · mean) | 16:44 실행과 같다 | 10:50 실행과 같다 |

    사슬은 여기서 멈췄다 — 가짜 모델 시험 · 스모크 · 기동은 시작하지 않았다.
  - **차단의 사후 진단 (CPU, 독립 검증 "확인" — 게이트가 통과한 것이 아니라 차단의 원인을 가린 것이다)**: HF forward 를 float32 로 CPU 에서 돌려 (순수 torch 대체 경로, `s4_step40/fp32ref/`) 반올림 없는 기준을 만들고 두 bf16 forward 의 거리를 쟀다. **움직인 쪽은 bf16 HF 참조다.**

    | KL(float32‖·) 프롬프트 0 · 1 · 2 | bf16 HF 참조 | Megatron |
    |---|---|---|
    | step 40 | 0.003624 · 0.000760 · 0.006821 | 0.000308 · 0.000083 · 0.000886 |
    | step 60 | 0.003700 · 0.000058 · 0.000713 | 0.001563 · 0.000293 · 0.000638 |

    float32 끼리의 step 40 대 60 (20 step 의 실제 가중치 변화) 은 0.000123 · 0.000028 · 0.000575 — bf16 HF 참조끼리 보이던 0.0004 · 0.0012 · 0.0077 은 대부분 참조의 반올림이다. 프롬프트 2 의 두 토큰은 float32 에서 41.546 · 41.777 이고 bf16 로 반올림하면 Megatron 의 값 (41.50 · 41.75) 그대로다. 검증자의 블라인드 1 단계도 저장된 로짓만으로 "HF step 40 참조가 다른 넷 (HF step 0 · 60, Megatron step 40 · 60) 과 동떨어져 있다" 를 독립으로 짚었고, 2 단계에서 float32 와 Megatron 의 차이 방향이 같음 (HF 참조 대비 변화의 상관 r = +0.958) 을 더했다. float32 CPU 경로는 Pai 에서 검증된 GPU 경로가 아니다 — 서로 다른 두 구현 (torch 대체 경로와 Megatron) 이 걸린 프롬프트에서 0.00008 로 일치한다는 것이 유효성의 근거다. HF bf16 쪽이 왜 움직였는지 (bf16 라우터 게이트 가설) 는 시험하지 않았다.
  - **부수 발견 (게이트의 한계)**: bf16 HF 참조 자체가 float32 에서 프롬프트 1 은 체크포인트에 따라 최대 0.0008, 프롬프트 0 은 늘 0.0036 떨어져 있다 — 빌린 KL 기준과 같은 크기라 M2 의 KL 조건은 참조의 오차에 묶인다. 참조를 float32 로 바꾸는 것은 게이트 변경이라 따로 검토한다 (지금은 하지 않는다).
  - **학습 경로의 정합 (이 런은 라우팅 재생을 끈다 — R1 이 아니라 아래가 지킨다)**: M3 0.0353 (step 60 은 0.0565) · `s4` 런 안의 step 36 ~ 46 `gen_kl_error` 0.0016 ~ 0.0019 · `token_mult_prob_error` 1.022 ~ 1.038 · 본 런 첫 step 의 KL 기준 0.002 (마지막 판정).
  - **사용자 결정 (2026-11-27 18:55, `RL_PLAN.md` 결정 96)**: step 40 을 시작점으로 수용한다. 사전에 정한 KL 조건은 차단으로 남는다 (통과로 고쳐 적지 않는다). 본 런 step 1 ~ 2 의 KL 이 0.002 를 넘으면 멈추고 보고한다.
- **GPU 스모크 `s6smoke` (2026-11-27 19:04 ~ 20:08, 4 × 8 · 1 step) — 통과하지 못함.** 벌점 기준은 모두 맞았다 (벌점 대상 15 = 턴 한도 8 · 문맥 한도 7, 보상 −1 · 손실에 남음 15/15, 문맥 상한 사례 1/1, advantage 재계산 32/32, KL 0.00137, 유효 32/32). 걸린 것: 배치 재시도 3 건 · 종료 때 vLLM 엔진의 `sem_unlink` Traceback 2 건. 원인은 Gym 의 컨테이너 격리 구멍 (`KNOWN_ISSUES.md` 2026-11-27) — 한 컨테이너의 삭제가 공유 `evaluation/oh` 와 main1 의 `/dev/shm` 을 지웠다. `s6` 은 띄우지 않았다.
- **격리 보강 (사용자 승인 2026-11-28 "격리보강은 권고대로 진행해") — Gym 패치 0004 + 점검 도구.** 도구는 `$NRL_ROOT/gates/swe_e2e_s6/`.

  | 무엇 | 도구 | 결과 (2026-11-28) |
  |---|---|---|
  | 지우지 않는 탐침 — 실제 Gym 경로 (가짜 모델), 에이전트 · 채점 컨테이너가 공유 위치마다 자기 파일을 만들어 보고 보이는 것을 센다 | `run_stub_probe.sh` + `stub_probe.py` + `summarize_probe.py` | **패치 전** (`probe_pre1` · `probe_pre2`): 호스트 `/dev/shm` 이 보임 (남의 항목 main1 50 · sub1 2,741) · `evaluation/oh` 쓰기 가능 + 다른 런 20 ~ 240 개 보임 · `logs` · `.eval_sessions` · 프롬프트 템플릿 쓰기 가능 · 채점 컨테이너의 `swebench_setup` 쓰기 가능. **패치 뒤** (`probe_post3`, 에이전트 9 + 채점 2 컨테이너, 두 노드): 전용 `/dev/shm` (남의 항목 0) · `evaluation/oh` 쓰기 불가 + 자기 런 1 개만 보임 (두 마운트 경로 모두) · 템플릿 · `swebench_setup` 쓰기 불가 · 호스트에 둔 표지가 어느 컨테이너에서도 안 보이고 끝난 뒤 그대로 · 양성 대조 (64 MB 쓰기 · python 공유 메모리 · 세마포어 9/9 · `mpirun` 2/2) 통과 |
  | 첫 시도의 실패를 탐침이 잡음 | `probe_post1` | `/dev/shm` 위에 롤아웃별 디렉토리를 붙이는 방법은 Apptainer 가 건너뛴다 ("source and destination must be identical when binding to /dev") — 판정 "문제 17 건". `--contain` 으로 바꿈 |
  | 회귀 — 미완 벌점 가짜 모델 시험 | `run_stub_unfinished.sh` on / off (`stubunf_on3` · `off3`) | 정답 패치 6/6 해결 · 턴 한도 6 개는 켬 −1 (마스킹 없음) · 끔 0 (마스킹) — 새 마운트에서도 채점이 된다 |
  | 빈 궤적 재실행 — 단위 | `test_rerun_loop.py` | 8/8. 패치를 뺀 사본에서는 실패 |
  | 빈 궤적 재실행 — 끝까지 (삭제 없이: 가짜 모델이 일부 과제의 첫 호출을 문맥 초과로 거절) | `run_stub_rerun.sh` on / off (`rerun_on1` · `off1`) | 켬: 거절한 6 과제가 한 번씩 다시 돌아 12/12 해결 · `Empty trajectory for` 6 줄. 끔: 6 과제의 출력이 빈 채 돌아옴 (문맥 한도 · 보상 −1 · 마스킹 없음 — 학습기에서는 배치 재시도가 되는 경로) |
  | 격리 사고의 흔적 점검 | `isolation_checks.py` (스모크 · 첫 step 점검기 · 감시가 함께 씀) | 자가 시험 15 건 + 감시의 증가분 규칙 6 건 통과. 사고가 난 `s6smoke` 에서 재시도 3 건 · 결과 파일 유실 12 건을 잡는다 |

  판정 기준 (점검 도구): 스모크 · 첫 step — 배치 재시도 · `Expected exactly one file` · `Empty trajectory for` 가 한 건이라도 있으면 실패, 호스트 표지 (`evaluation/oh` · `logs` · 두 노드 `/dev/shm`) 가 사라지면 실패, 호스트 Shmem 이 500 GB 를 넘으면 실패. 감시 (본 런) — 표지 유실 · Shmem 500 GB 초과 · 지난 점검 뒤 유실 + 재실행 3 건 이상 증가는 중단 조건, 1 ~ 2 건 증가 · Shmem 200 GB 초과는 보고. 200 · 500 GB 와 3 건은 잡음 바닥이 아니라 안전 한계로 고른 값이다.
  **적대적 검토** (같은 검증자): 패치는 타당. 점검은 두 번 "보완 필요" → 반영 뒤 스모크 쪽 "신뢰 가능". 지적과 반영: ① 재실행이 성공하면 롤아웃 사이로 번진 삭제가 점검을 통과한다 → 유실 · 재실행 줄을 실패 기준으로, 표지를 도입 ② 탐침이 두 마운트 경로 중 하나만 봤고 "1 개만 보임" 이 빈 디렉토리로도 통과한다 → 호스트 경로 별칭과 표지 가시성 추가 ③ `/dev/shm` 항목 수 비교는 늘어난 항목에 가려진다 → 표지로 대체 ④ SWE-Gym 채점 컨테이너의 `swebench_setup` 은 여전히 쓰기 가능 → 이 런의 데이터는 7,002/7,002 SWE-rebench, 실행기가 다른 종류가 섞이면 멈춘다 ⑤ 감시의 누적 기준은 며칠 안에 반드시 넘는다 → 증가분 기준.
  **GPU 스모크 `s6smoke2` (2026-11-28 10:00 ~ 10:45, 8 × 8 · 1 step · `teacher_swe_e2e_smoke_ctx16.jsonl` 16 과제 · 시작 `hf_s4/hf_step_00040` · commit `a354d8579`) — 통과.** 과제를 16 개로 늘린 까닭: 데이터가 섞여 들어가 4 × 8 로는 "문맥 상한 사례 1 개 이상" 이 우연히 안 채워질 수 있고 (`s6smoke` 에서 1 개), 동시에 도는 두 배치가 과제를 나눠 갖지 않게 하려는 것.

  | 기준 | 결과 |
  |---|---|
  | 벌점 대상이 보상 −1 · 손실에 남음 · 하니스 표지 | 23/23 (턴 한도 15 · 문맥 한도 8), 문맥 상한에서 끝난 것 2/2 |
  | 대상이 아닌 롤아웃 | 보상 0 이 34 · 1 이 7, 마스킹 1 (지표와 일치), 풀었는데 보상 0 인 것 0 |
  | advantage 재계산 | 64/64 일치 (벌점 대상 평균 −1.25 · 끝까지 간 실패 평균 +0.58) |
  | KL (기준 0.002) | 0.00143, 유효 시퀀스 63/64 |
  | 격리 | 배치 재시도 0 · 결과 파일 유실 0 · 빈 궤적 재실행 0 · 표지 4/4 유지 · 호스트 Shmem 최대 44 · 20 GB |
  | Traceback | 학습기 쪽 0 · 하니스 롤아웃 쪽 0 |
  | 첫 step 점검기 (`check_g1.py`) | 통과 — 짝지은 롤아웃 64/64 · 거부 표지 51 vs 지표 51 |

  에이전트 명령 실패는 128 롤아웃 중 3 건 (2.3% — `s4` 3.1% · `s5` 9.1% 보다 낮다, `--contain` 의 부작용으로 볼 것은 없었다. 16 과제의 표본이다). *정정 (2026-11-28 12:45)*: 처음 적은 비교값 `s4` 2.1% · `s5` 4.2% 는 로그 줄 수였다 — Gym 서버 로그는 같은 꼴의 줄을 `[repeated Nx across cluster]` 로 접어서 줄 수가 실제 건수보다 적다. 위 값은 롤아웃별 지표 파일 (`nemo_gym_metrics.json` 의 `agent_timed_out`) 로 다시 센 것이다 (`s4` 124/3,968 · `s5` 93/1,024).
  **남는 위험 (고치지 않고 적는다)**: 제 결과를 지운 롤아웃은 한 번 다시 돌고 또 비면 배치 재시도가 된다 · 첫 프롬프트가 문맥을 넘는 과제는 다시 돌아도 빈다 · 다시 돈 롤아웃의 첫 시도는 손실에 들어가지 않는다 (자기 환경을 망가뜨린 행동에 벌점이 없다) · 에이전트 명령이 실패한 롤아웃은 마스킹 없이 보상 0 (전과 같다, 결정 대기) · 에이전트가 `/trajectories_mount` 의 지표 파일을 고쳐 제 롤아웃을 마스킹할 수 있다 · `--contain` 의 최소 `/dev` (`/dev/fuse` · `/dev/full` 없음) 가 과제 테스트에 주는 영향은 탐침한 9 개 이미지 밖에서 재지 않았다 · 전용 `/dev/shm` 상한 약 1 TB (Shmem 감시로 본다) · 호스트 `/proc` · 네트워크 · opencode 경로.
- 결과는 이 절 아래에 적는다.

**본 런 `teacher_swe_e2e/s6` step 1 ~ 2 (2026-11-28 10:53 기동 · step 1 12:39 · step 2 12:52, commit `8603802b7`) — 결정 96 의 마지막 판정 통과 (KL 0.00162 · 0.00154, 기준 0.002 미만).**

| 항목 | step 1 | step 2 | 견줄 값 |
|---|---|---|---|
| Generation KL (기준 0.002 미만) | 0.00162 | 0.00154 | `s4` step 31 ~ 40 평균 0.00169 |
| 미완 벌점 (보상 −1, 손실에 남음) | 15.2% (턴 9.4 · 문맥 5.9) | 35.5% (턴 17.2 · 문맥 18.4) | `s5` step 1 · 2 에서 한도에 걸린 비율 29.7% · 45.3% (그때는 마스킹) |
| 해결률 | 3.1% (8/256) | 0.8% (2/256) | `s5` 4.7% · 0.0% |
| 보상 평균 | −0.121 | −0.348 | — |
| 마스킹 (유효 시퀀스) | 22.7% (198/256) | 7.4% (237/256) | `s5` 44 ~ 50% (한도에 걸린 것 포함) |
| 시간 초과 (마스킹되는 것) | 22.3% (57) | 6.6% (17) | `s5` 14.5% · 3.9% · 18.0% (셋째 배치) |
| logprob 필터로 마스킹 | 1 | 1 | `s4` step 1 ~ 40 가운데 6 개 step 에서 1 ~ 2 (step 당 64 롤아웃) |
| 패치를 낸 비율 (전체) | 62.1% | 71.5% | `s4` step 2 ~ 10 평균 76.9% |
| 끝까지 간 실패: 패치 · 턴 중앙 · 20 턴 미만 | 80.3% · 101 · 2.0% | 80.7% · 108 · 4.1% | `s5` 81.7% · 99 · 2.3% / 88.4% · 105 · 2.3% |
| 턴 수 평균 | 120.8 | 135.5 | `s4` step 2 ~ 10 평균 131.8 |
| 격리 | 배치 재시도 0 · 결과 파일 유실 0 · 빈 궤적 재실행 0 · 표지 4/4 · 호스트 Shmem 최대 27 · 18 GB | 같음 | 중단 기준: 표지 유실 · 3 건 이상 · 500 GB |
| W&B | `_step` 41 | `_step` 42 | `s4` 의 1 ~ 40 뒤에 이어짐 (42 행, 빠진 step 없음) |

- **G1 (`check_g1.py --steps 2`, 2026-11-28 12:52 ~ 13:14, `$NRL_ROOT/gates/swe_e2e_s6/s6/g1.txt`): 실패 (rc 1) — 기준 둘 미달, 정합 항목은 전부 통과.**

  | 항목 | step 1 | step 2 | 판정 |
  |---|---|---|---|
  | 짝지은 롤아웃 (학습기 조립 재현) | 256/256 | 256/256 | 통과 |
  | 거부 표지 (조립 재현 vs 학습기 지표) | 99 = 99.0 | 89 = 89.0 | 통과 |
  | 이력 있는 롤아웃의 거부 호출 확인 · 버려진 턴 · 표지 누락 · 거짓 표지 | 84/87 · 0 · 0 · 0 | 82/82 · 0 · 0 · 0 | 통과 |
  | 미완 벌점 대상 (턴 · 문맥) vs 지표 | 39 (24 · 15) = 39.0 | 91 (44 · 47) = 91.0 | 통과 |
  | 마스킹 대상 vs `num_mask_sample_filtered` · 풀었는데 보상 0 | 57 = 57.0 · 0 | 18 = 18.0 · 0 | 통과 |
  | KL (0.002 미만) · 유효 시퀀스 (128 이상) | 0.00162 · 198 | 0.00154 · 237 | 통과 |
  | **하니스 이력 없는 롤아웃 (상한 12.5%)** | **58/256 (22.7%) — 시간 초과 57 · 빈 이력 1** | 21/256 (8.2%) | **step 1 미달** (`s5` 도 step 1 에서 38/256 = 14.8% 로 미달) |
  | **`num_masked_seqs_by_logprob_error` (기준 0)** | **1** | **1** | **미달** (`s5` 0 · 0, `s4` 60 step 중 9 step 에서 1 ~ 2) |

  정의는 "어긋나면 중단 · 진단 · 보고" 다. 런은 멈추지 않았다 — 걸린 롤아웃은 둘 다 손실에서 빠지는 것이고 (마스킹 · 필터), 중단은 사용자 결정 사항이다. 진단 (`problem-diagnosis`) 과 독립 검증의 판정은 아래에 잇는다.
  **진단 · 독립 검증 (13:15 ~ 13:50): 두 항목 모두 "확인 (수정 포함)"** — 서사와 반론은 `KNOWN_ISSUES.md` (11-28).
  - 이력 없는 롤아웃: 롤아웃 128 개가 한꺼번에 뜨는 물결 (기동 직후 · 시스템이 빈 뒤의 셋째 배치) 에서 기동이 중앙 약 600 초 걸려 1,800 초 한도의 루프 시간이 줄고, 평소 길이의 긴 꼬리가 걸린다. 패치 전 `s5` 에도 같은 물결 효과가 있다 (동시 21% · 34% 대 엇갈림 0 ~ 8%). step 3 도 41/256 (16.0%). `s6` step 1 이 `s5` 보다 많은 것 (57 대 37, 같은 과제 · p 0.023) 은 step 1 에 한정되고 (step 3 은 41 대 46) 약 9 건을 가르지 못했다.
  - logprob 필터: `s4` 의 기저율 (대상 시퀀스 2,488 개 중 10 개 = 0.40%) 이면 대상 199 · 238 개인 step 의 55% · 62% 에서 1 개 이상이 나온다.
  - **두 기준은 step 당 64 롤아웃 때의 값이라 256 롤아웃의 물결 step 에서는 정상 상태에서도 깨진다** — 기준 보정은 측정 방식 변경이라 승인 · 적대적 검토 대상이다 (결정 대기). 그 전까지 G1 의 공식 판정은 "정합 항목 통과 · 기준 둘 미달 · 원인 진단 확인" 이다.
  - 영향: 걸린 롤아웃은 손실에서 빠졌다. 마스킹된 시간 초과는 보상 0 으로 그룹 기준선에 남아 (upstream 설계) 벌점이 든 그룹 9 ~ 13 개에서 남은 롤아웃의 advantage 를 평균 0.21 ~ 0.27 옮겼다 (부호 반전 0 · 절댓값 합 +2.1 ~ 2.3%). 되돌릴 근거는 없다 (검증자 판단).
  - 점검기 자체의 비용: 롤아웃마다 30 ~ 70 MB 인 마지막 호출 덤프를 읽어 512 롤아웃에 22 분 · NFS 약 45 GB 다. 런 중에 도는 점검으로는 무겁다 (보정 대상).
- 감시 (`monitor.py`) 판정은 두 step 모두 "보고할 것 있음" (중단 조건 없음). 보고 항목 셋을 `s4` · `s5` 와 견줬다 (읽기 전용):
  - **시간 초과 22.3% (step 1)**: 에이전트 명령 실패 74 건 (512 롤아웃, 14.5%) 은 전부 반환 코드 124 (제한 시간) 다 — 새 실패 종류는 없다. 롤아웃 시간 구성은 `s5` 와 같다: 컨테이너 기동 중앙 0.7 초 (같음) · 명령 실행 173 초 (`s5` 180) · 모델 호출 545 초 (597) · 런타임 연결 133 초 (`s5` 첫 배치 81 · 셋째 배치 159) · 둘째 배치는 전 항목이 같다 (연결 13.1 대 13.2 초). 첫 물결에 128 개 컨테이너가 한꺼번에 뜨며 런타임 연결 · 초기화가 느려지는 기존 현상이다 (`E2E_SANDBOX.md` §7.7). 패치 비율 62.1% 는 시간 초과 몫이 커서 내려간 값이고, 끝까지 간 롤아웃의 패치 비율은 80% 로 `s5` 와 같다.
  - **하니스 Traceback 3 줄**: 롤아웃 2 건 — 런타임 서버가 연결 중 죽음 (`Server process died`, 1 건) · 초기 설정 명령 시간 초과 (`Failed to run initial setup`). 에이전트 로그 전수로 세면 초기 설정 실패는 `s5` 10/1,024 · `s6` 9/687, 서버 죽음은 `s5` 0 · `s6` 1 이다.
  - **오류 종류 "other" 7 건 (1.4%)**: JSON 해석 오류 5 · 연결 끊김 2 — `s4` 에도 있던 두 종류다 (`s4` 33/3,968 = 0.8%: JSON 23 · 연결 끊김 2 · 그 밖 8).
- `--contain` 의 부작용으로 볼 것은 step 1 ~ 2 에서 찾지 못했다 (새 실패 종류 0 · 기동 시간 같음). 시간 초과 비율은 계속 본다 — 첫 물결 뒤 배치에서도 `s5` 셋째 배치처럼 18% 가 나온 적이 있다.
- 감시 도구 약점 (2026-11-28 발견, 고치지 않음 — 측정 방식 변경이라 승인 대기): 격리 점검이 Gym 서버 로그의 **줄 수**를 센다. Ray 가 5 초 안에 반복된 같은 꼴의 줄을 `[repeated Nx across cluster]` 한 줄로 접으므로 `Expected exactly one file` 이 한꺼번에 터지면 실제보다 적게 센다 (에이전트 명령 실패: 줄 47 · 실제 74). 0 이 아닌 값은 항상 한 줄 이상 남아 보고는 되지만, "3 건 이상 → 중단" 이 "1 ~ 2 건 → 보고" 로 읽힐 수 있다. 표지 유실 (중단 조건) 은 영향 없다. `Empty trajectory for` 는 서버 프로세스가 직접 찍어 접히지 않는다.

**본 런 `s6` step 3 ~ 5 (2026-11-28 13:52 · 14:30 · 15:11, 첫 체크포인트 `ckpt_s6/step_5` 153 GB · 15:10 저장)** — 감시 판정 "보고할 것 있음", 중단 조건 없음.

| step | KL | 엔트로피 | 생성 토큰 | 턴 | 턴당 최대 생성 | 패치 | 미완 벌점 | 해결률 | 마스킹 | 시간 초과 | logprob 필터 | 주기 (초) | lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00162 | 0.399 | 28,111 | 120.8 | 2,309 | 62.1% | 15.2% | 3.1% | 22.7% | 22.3% | 1 | — | 1e-7 |
| 2 | 0.00154 | 0.378 | 31,405 | 135.5 | 3,183 | 71.5% | 35.5% | 0.8% | 7.4% | 6.6% | 1 | 660 | 1.07e-6 |
| 3 | 0.00156 | 0.385 | 28,362 | 125.1 | 2,898 | 70.3% | 20.7% | 2.0% | 16.0% | 16.0% | 0 | 3,727 | 2.03e-6 |
| 4 | 0.00155 | 0.370 | 29,904 | 133.9 | 3,052 | 68.0% | 23.8% | 3.9% | 13.3% | 12.1% | 1 | 2,222 | 3e-6 |
| 5 | 0.00150 | 0.357 | 29,446 | 128.1 | 3,337 | 79.3% | 28.9% | 3.1% | 6.6% | 5.9% | 2 | 2,532 | 3e-6 |

- 격리: 배치 재시도 0 · 결과 파일 유실 0 · 빈 궤적 재실행 0 · 표지 4/4 (step 5 까지).
- 진단의 예측 ("넷째 배치부터는 128 개 동시 시작이 없어 시간 초과가 엇갈린 물결 수준") 은 step 4 · 5 에서 12.1% · 5.9% 로 맞았다.
- 감시의 G4 보고 "턴당 최대 생성 이동평균 1.46 배": 기준선이 다른 런 (`s4` step 2 ~ 10, 8 × 8 · 평균 2,190) 이다. 같은 가중치로 생성한 `s6` step 1 과 2 가 2,309 · 3,183 으로 갈리므로 과제 구성의 차이가 크다. `s6` 자체의 step 1 ~ 3 평균 2,796 대비로는 1.14 배다. 추세로 볼 것은 엔트로피다 — 0.399 → 0.357 (5 step 에 −10%, `s4` 의 step 값 범위 0.36 ~ 0.43 의 아래쪽 끝). G4 의 ±30% 안이고 계속 본다.
- logprob 필터는 5 step 에 5 건 (대상 시퀀스 1,116 개의 0.45% — `s4` 0.40%).
- 끝나는 모습 (`behaviour.py`): 끝까지 간 실패의 패치 비율 80 ~ 86% · 20 턴 미만 0.7 ~ 2.0% (벌점 없던 `s5` 2.3%) — 일찍 포기하는 쪽으로의 변화는 보이지 않는다.
- 학습 노드 (sub1) 의 공유 메모리가 step 5 저장 때 15 → 172 GB 로 올라 유지된다 — 정책 워커 8 개가 20 ~ 25 GB 씩 잡은 것이고 (`RssShmem`) `/dev/shm` 파일은 3.6 GB, 컨테이너와 무관하다 (롤아웃은 main1). `s4` 에서도 저장 때 그 노드의 호스트 메모리 최대가 211 GB 였다 (평소 15 ~ 17 GB). 감시의 보고 기준 200 GB · 중단 기준 500 GB 는 컨테이너가 호스트 공유 메모리를 채우는 것을 잡으려는 값이다 — step 10 저장에서 누적되는지 본다.
- step 주기: step 4 · 5 가 2,222 · 2,532 초 (약 40 분). 남은 213 step 이면 약 5.8 일 — 기동 전 추정 (step 당 68 분 · 약 10 일) 보다 짧다.

**본 런 `s6` step 6 ~ 10 (2026-11-28 15:45 ~ 18:29, 둘째 체크포인트 `ckpt_s6/step_10` · HF 반출 `hf_s6/hf_step_00005` 15:20 — 텐서 14,181 · 오류 0 · 동결 텐서 변화 0)** — 감시 판정 "보고할 것 있음", 중단 조건 없음.

| step | KL | 엔트로피 | 생성 토큰 | 턴 | 패치 | 미완 벌점 | 해결률 | 마스킹 | 시간 초과 | logprob 필터 | 주기 (초) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.00158 | 0.388 | 27,771 | 126.4 | 78.1% | 27.7% | 1.2% | 9.4% | 8.6% | 1 | 1,944 |
| 7 | 0.00161 | 0.395 | 28,915 | 131.1 | 61.7% | 18.0% | 0.0% | 21.5% | 21.1% | 1 | 2,632 |
| 8 | 0.00158 | 0.377 | 26,373 | 123.4 | 76.2% | 29.7% | 0.0% | 5.5% | 5.5% | 0 | 2,013 |
| 9 | 0.00156 | 0.375 | 28,133 | 129.7 | 79.7% | 27.3% | 6.6% | 7.0% | 6.2% | 2 | 2,178 |
| 10 | 0.00160 | 0.397 | 25,788 | 119.7 | 77.0% | 21.9% | 5.1% | 12.5% | 11.3% | 1 | 3,040 |

- 격리: 배치 재시도 0 · 결과 파일 유실 0 · 빈 궤적 재실행 0 · 표지 4/4 (step 10 까지).
- **G1 진단의 예측 하나가 틀렸다 — 동시 기동은 넷째 배치 뒤에도 작게 되풀이된다.** 예측은 "step 4 이후 시간 초과 1 ~ 16%" 였는데 step 7 이 21.1% 다. 시작 시각별로 보면 대기열이 비었다가 새 배치가 들어올 때 60 ~ 90 개가 한꺼번에 뜬다: 15:05 ~ 15:10 에 89 개 (기동 지연 중앙 219 · 145 초, 시간 초과 34% · 29%) · 16:25 ~ 16:30 에 120 개 (183 · 145 초, 8 ~ 9%) · 17:00 ~ 17:10 에 130 개 (154 ~ 187 초, 10 ~ 17%). 15:05 의 것은 step 5 의 저장 (15:07 ~ 15:10) 과 겹친다 — 학습 + 저장이 길어져 돌던 배치가 먼저 끝난 것으로 읽히지만 확인한 것은 시각의 겹침뿐이다. step 1 ~ 10 의 마스킹은 평균 11.8% (302/2,560) 이다.
- 마스킹된 롤아웃이 그룹 기준선에 남는 효과를 step 1 ~ 10 으로 다시 쟀다 (`diag_g1/mine/adv_steps.py`, step 1 은 검증자의 값과 같다): step 마다 남은 롤아웃 65 ~ 105 개 (약 30%) 의 advantage 가 평균 0.14 ~ 0.27 달라지고 (남은 롤아웃의 advantage 절댓값 평균은 약 0.7), 부호 반전은 0, 절댓값 합은 0.996 ~ 1.046 배 (합계 1.023 배) 다. 벌점이 없던 `s5` 는 704 롤아웃에서 6 개만 달라졌다. step 이 지나며 커지는 추세는 없다.
- 엔트로피는 0.357 (step 5) 뒤 0.375 ~ 0.397 로 돌아왔다 — 하락 추세가 아니었다. 턴당 최대 생성의 이동평균도 1.46 배 → 1.17 배.
- logprob 필터: 10 step 에 10 건 (대상 약 2,300 개의 0.43% — `s4` 0.40%).
- 학습 노드 공유 메모리: 둘째 저장 뒤에도 171 ~ 186 GB — 누적되지 않는다. 호스트 메모리는 저장 뒤 314 ~ 396 GB 에서 평평하다 (30 분 구간의 최소 ~ 최대, 15:30 부터).
- 끝나는 모습: 끝까지 간 실패의 패치 비율 79 ~ 89% · 20 턴 미만 1.3 ~ 6.9% (step 6 만 6.9%, `s5` 2.3%) · 턴 중앙 94 ~ 106.
- step 주기 (6 ~ 10): 평균 2,361 초 (39 분). 남은 208 step 이면 약 5.7 일.

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
