# alpha RLVR 이식 준비 — 위험 보고 · 게이트 계획

2026-10-07 위험 보고(사용자 요청: "Pai 의 중요한 학습 config 를 NeMo-RL 에서 재현하기 전에 잠재 위험을 보고")의 정본이다.
게이트 G0~G7 의 진행 상태는 [`STATUS.md`](STATUS.md), 수치 결과는 [`GATES.md`](GATES.md) 에 쓴다. 사용자 승인: 기록·G0·G1 (2026-10-07).
**계획 재편 (사용자 지시 2026-10-07)**: 첫 목표는 **정확한 RLVR 학습**이다. 속도 최적화는 실제 RLVR 런에서 학습·롤아웃 중 어느 쪽이 병목인지 확인한 뒤에 한다 (§6).

**검증 방법 (2026-10-07)**: 코드 정적 분석(병렬 감사 4건 — GDN packing·CP, R3 호환, loss·clip·MoE 학습 의미, 브리지·vLLM 128K — 핵심 주장은 직접 재확인),
RLVR1 블렌드(`rlvr1_alpha.jsonl`) 99,113행 CPU 집계, R4 산출물 점검. GPU 실행은 하지 않았다.
경로 표기: mcore = `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`(워커 venv 가 editable 로 쓰는 `d12f6c8c9`).

## 1. 결론

1. **모델 상수·Muon 동역학·loss 단위는 Pai SFT 와 같다.**
   - 브리지 상수를 Pai `baseline_48L.yaml` 과 전수 대조했다(변환 캐시 `run_config.yaml`). 전부 일치한다. M1·M2·M5·R2 도 통과했다.
   - SFT 의 `clip-grad 8.0`(CP8)은 NeMo-RL 의 `max_grad_norm: 1.0` 과 같은 크기다. Pai 는 grad norm 이 CP 배수로 기록되고 NeMo-RL 은 CP 와 무관하다.
2. **가장 큰 위험은 검증 범위다.** 지금까지의 PASS 는 4K 이하·CP1·packing 없음·1노드·sync·단일 턴 경로에서만 나왔다.
   128K RLVR 은 packing+CP8·async·Gym 멀티턴·2노드 경로를 쓴다. 이 경로의 KL·R3 수치는 아직 없다.
3. **당장 막을 함정은 셋이다.**
   - Pai 의 `moe-router-fusion: true` 를 옮기면 R3 가 에러 없이 꺼진다 (H1).
   - Gym 서버의 도구·추론 파서를 alpha 용으로 지정하지 않으면 도구 호출이 벌점이 된다. 블렌드 행의 44% 가 도구를 쓴다 (H3).
   - RLVR1 블렌드의 23.4% 는 judge 모델이나 코드 sandbox 가 있어야 채점된다. 2노드 계획에는 그 자리가 없다 (H4).
4. **128K 학습 처리량은 Pai SFT 의 약 1/1.9 다** (정상 상태 33.4K vs 63.6K tok/s, R5 2026-10-07). 첫 보고의 "1/4"는 R4 가 워밍업이 덜 끝난 2번째 스텝을 잰 값이었다.
   SFT 최적화 중 router fusion 은 R3 와 충돌한다. 속도 최적화는 첫 RLVR 런 뒤로 미룬다 (§5).

## 2. Pai SFT 핵심 설정 → NeMo-RL 대응

| 항목 | Pai SFT (`sft_128k_agentic.yaml`·`baseline_48L.yaml`) | NeMo-RL 현재 (alpha 레시피 + `grpo_math_1B.yaml` 상속) | 판정 |
|---|---|---|---|
| 모델 상수 (층·차원·GQA·GDN·RoPE θ 1e7·partial 0.25·표준 RMSNorm·QK-norm·gate·vocab 163,968) | 위 | AlphaBridge provider | 일치 |
| DSV3 라우팅 (sigmoid·8그룹 top-4·top-8·scaling 2.5·expert_bias fp32) | 위 | provider | 일치 |
| 라우터 연산 dtype | fp32 | fp64 (상속값이 provider 를 덮음, `nemo_rl/models/megatron/setup.py:861`) | 다르지만 무해 |
| 라우터 가중치 | 학습 | 동결 (`freeze_moe_router`, `setup.py:1540`) | 의도적 차이 |
| 부하 균형 | `seq_aux_loss` 1e-4 + bias 갱신 1e-3 | 없음 + 갱신 0.0 (Ultra 는 bias 1e-3) | 다름 → D3 |
| Muon 하이퍼파라미터 | `dist_muon`·nesterov·NS 5·spectral·extra_scale 0.2·β2 0.95·wd 0.1·QK-norm wd | `grpo_alpha_smoke_muon.yaml` | 일치 (R2) |
| lr·스케줄 | 1e-5 constant, warmup 50 optimizer step | 1e-6, warmup 0 (KL 게이트용). 스케줄 1 tick = GRPO step 1회 (`megatron_policy_worker.py:989-995`) | 결정 → D5 |
| grad clip | 8.0 @CP8 | 1.0 | 같은 크기. SFT 에서는 한 번도 안 걸렸다 (Pai agentic 로그 1,337 iter, CP 정규화 p50 0.012·최대 0.17) |
| loss 정규화 | bin 별 토큰 평균, grad norm ∝ CP | 전역 토큰 평균, CP 무관 (`nemo_rl/models/megatron/train.py:507-525`) | 단위 같음. Pai 로그는 ÷CP 해서 비교 |
| grad reduce | fp32 | Muon 레시피 true, Adam 스모크 false | Muon 일치 |
| 128K 분할 | CP8 + THD 문서 격리 | CP>1 은 packing 필수 (`setup.py:840`). GDN 상태 리셋은 코드상 정상 | alpha 실측 없음 → H2 |
| recompute | selective `layernorm moe` | full, 1층 uniform (상속 `recompute_granularity: "full"`) | 다름. R4 의 128K OK 는 full 기준 → §5 |
| 옵티마이저 상태 offload | chunked offload (Pai 기능 #4) | 없음. R4 고정 바닥 27 GB (LayerWise 가 dp_cp 로 샤딩) | §5 에서 재평가 |
| 할당자 | `expandable_segments:True` 항상 (Pai `train.sh:82`) | 미설정 | 다름 → D6 |
| MoE 융합 | grouped gemm·permute·router fusion | grouped gemm·permute 같음, router fusion 꺼짐 | router fusion 은 켜면 안 됨 → H1 |
| rope 융합 | 켜짐 | 꺼짐 (사유 기록 없음) | 성능 차이 → §5 |
| gradient accumulation fusion | 켜짐 | 꺼짐 (`setup.py:1037`) | 영향 낮음 |
| `NCCL_MAX_NCHANNELS=16` | 옵티마이저 상태 재개 OOM 회피 (Pai 2026-07-15) | 미설정 | G7 재개(2노드·CP8)에서 OOM 없음 — 미설정 유지 |
| FlashQLA | 채택 보류 (CP4/32K 는 fla 우세, 128K 형상만 2.2×) | opt-in | packed 다중 시퀀스 미검증 → fla 유지 |
| effort 마커 | 템플릿이 `\n\n{reasoning effort: efficient}` 부착 | 블렌드 3,429행(3.5%)이 같은 형식. `env.nemo_gym.effort_levels` 미설정 | 형식 일치. 계수는 Ultra 레시피에 공개 (0.1/1/15000) |

## 3. 검증된 경로와 128K RLVR 경로

| 축 | 게이트가 본 조건 | 128K RLVR 조건 |
|---|---|---|
| 길이 | R1 ≤4,096 · M5 2,048 | ≤131,072 |
| CP · packing | CP1 · 끔 | CP8 · 켬 (필수) |
| 토폴로지 · 동기 | 1노드 colocated · sync | 2노드 분리 · async + in-flight 갱신 |
| 롤아웃 경로 | 네이티브 단일 턴 (수학) | Gym HTTP 멀티턴 · 도구·추론 파서 |
| vLLM | eager · prefix cache 켬 (GDN 적중은 거의 없었을 것) | CUDA graph · prefix cache 결정 필요 |
| R3 | R1 켬 · R4 끔 | 켬 |
| 데이터 | R1 수학 · R4 랜덤 토큰 1개 | 22개 환경 혼합 |

R4 결과 JSON 의 `loss: NaN` 은 수치 문제가 아니다. 하네스가 `loss` 키를 읽는데 워커는 `global_loss` 를 반환한다
(`megatron_policy_worker.py:1005`). 그래서 R4 는 128K 에서 loss 가 유한한지 확인하지 못했다 (`KNOWN_ISSUES.md` 2026-10-07 항목, G0 에서 수정).

## 4. 위험

### 높음

**H1. router fusion 을 켜면 R3 가 조용히 꺼진다.**
- mcore fused top-k 는 replay 전에 반환한다 (mcore `megatron/core/transformer/moe/moe_utils.py:828-847`, replay 는 `:887-893`).
- `precomputed_indices` 에는 assert 가 있지만 `router_replay` 에는 없다. NeMo-RL 의 R3 설정 검증도 이 조합을 막지 않는다.
- 조치: R3 를 켠 레시피에서 `moe_router_fusion` 금지. 실행 전 검사 도구로 강제한다 (G0).

**H2. packing + CP8 + R3 경로를 alpha 로 실행한 적이 없다.**
- GDN 은 코드상 맞다. conv 와 delta-rule 이 둘 다 cu_seqlens 를 받는다 (mcore `megatron/core/ssm/gated_delta_net/gdn.py:214-252`).
  torch conv 경로는 deterministic 모드 전용이고 packing 과 함께 쓸 수 없게 막혀 있다 (`gdn.py:95-99`).
- THD+CP 순열은 NeMo-RL 의 CP 배치와 일치한다. R3 route 도 같은 경계로 패딩·분할된다.
- upstream 수치 테스트는 장난감 크기·CP≤2 뿐이다.
- "packing 끔" 규약은 과거 사정이다. Qwen3.5 레시피가 mcore 의 GDN packing 지원(2026-05-14) 이전에 만들어졌다.
- 조치: G1 (packing 상태 누출) · G2 (CP sweep + R3 trace + packing+CP R1).

**H3. Gym 서버 파서가 alpha 형식과 맞지 않으면 도구 호출이 벌점이 된다.**
- RLVR1 블렌드 99,113행 중 43,861행(44%)이 도구를 선언한다.
- Gym 학습 설정은 `uses_reasoning_parser: true` 다 (Gym `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`).
  NeMo-RL HTTP 서버는 `tool_parser`·`reasoning_parser` 를 지정하지 않으면 파서가 없다 (`nemo_rl/models/generation/vllm/vllm_worker_async.py:660-683`).
- Ultra 는 `qwen3_coder` + `nano_v3` 다. Pai 벤치 fleet 은 alpha 에 `qwen3_xml` + `nemotron_v3` 를 쓴다.
- 파서가 alpha 의 XML 도구 호출을 못 읽으면 `<tool_call>` 이 본문에 남는다. NeMo-RL 은 이를 invalid tool call 로 판정한다 (`nemo_rl/environments/nemo_gym.py:169-224`).
  Ultra 의 `invalid_tool_call_advantage: -5.0` 을 복사하면 정상 호출마다 −5 가 붙는다.
- `<think>`·`</think>` 는 alpha 토크나이저의 special token 이다. 추론 파서 없이 디토크나이즈하면 태그가 사라지고 추론이 답변에 섞인다 (Pai 08-30 사고 계열).
- R3 렌더 게이트는 vLLM 서버를 띄우지 않았다.
- 조치: G3.
- **G3 결과 (2026-10-07)**: 파서는 맞았다(invalid 0/64). 대신 다른 형태로 터졌다 — vLLM 0.25.1 이 `strict: true` 도구에 xgrammar 제약 디코딩을 걸어
  도구 롤아웃이 off-policy 가 됐다(KL 0.0045, 도구 시퀀스 1/3 마스킹). `VLLM_ENFORCE_STRICT_TOOL_CALLING=0` 으로 해결(KL 0.0017).

**H4. RLVR1 블렌드의 23.4% 는 현재 토폴로지에서 보상을 낼 수 없다.**

| 필요한 자원 | 환경 (`agent_ref.name`) | 행 | 비율 |
|---|---|---|---|
| GenRM 모델 | `genrm_simple_agent`(+`_reasoning_off`), `abstention_simple_agent` | 9,509 | 9.6% |
| LLM judge | `multichallenge_simple_agent` (기본값은 정책 자신이 채점) | 4,028 | 4.1% |
| 안전 judge (Nemotron-Content-Safety-Reasoning-4B) | `jailbreak_*` 4종 | 2,861 | 2.9% |
| 코드 sandbox | `ns_tools`(파이썬), `math_formal_lean` | 6,819 | 6.9% |
| 라이선스 미확인 | `nvarc` 2종 (ARC-AGI) | 4,166 | 4.2% |

- identity 주입 689행은 전부 `genrm_simple_agent` 다. GenRM 없이는 identity RL 신호가 없다.
- Ultra 는 judge 전용 노드 20대를 쓴다. alpha 2노드 계획(`RL_PLAN.md` §3)은 judge 를 MOPD 단계에서만 다룬다.
- `code_gen`(8.0%)은 sandbox 없이 Ray 워커에서 모델 코드를 실행한다 (Gym `resources_servers/code_gen/app.py:171`). 공유 노드라 격리를 검토한다.
- 데이터·자원 결정이라 승인 후 진행한다 → D1.

**H5. Ultra 의 `reward_penalties.token_ids` 는 Nemotron 토크나이저 id 다.**
- Ultra 값 `unwanted [2]`·`think_open 12`·`think_close 13` 은 alpha 에서 `<|im_start|>`·`<tool_response>`·`</tool_response>` 다. alpha 의 `<think>`·`</think>` 는 14·15 다.
- 복사하면 think 태그 검사가 도구 응답 태그를 보게 되어 보상이 조용히 오염된다.
- 조치: 레시피 골격은 alpha id 를 쓴다. `tools/check_alpha_recipe.py` 가 토크나이저와 대조한다 (G0).

### 중간

| # | 위험 | 내용 | 조치 |
|---|---|---|---|
| M1 | prefix caching 이 숨은 기본값으로 켜진다 | 키가 없으면 sm≥8 에서 켠다 (`vllm_worker.py:80-84`). vLLM 0.25.1 은 하이브리드 모델에 기본으로 끈다. 켜면 chunked prefill 이 강제되고, 이 조합에서 route 행이 드물게 누락된다 (`nemo_rl/models/generation/vllm/utils.py:195`). upstream R3 레시피 2종은 둘 다 끈다 | D2 |
| M2 | async(Gym) 경로의 vLLM 배치 기본값이 작다 | `AsyncLLM.from_engine_args` 를 usage_context 없이 부른다 (`vllm_worker_async.py:198`). 정적 판독상 batched tokens 2048·seqs 128 이다. 128K 프롬프트 하나의 prefill 이 약 124 스텝이다 | `max_num_batched_tokens`·`max_num_seqs` 명시 |
| M3 | 롤아웃 병렬은 TP=EP 만 가능하다 | async 는 EP≠TP 를 막는다 (`vllm_generation.py:134-141`). TP1 이면 GPU 당 가중치 30.2 GB, 128K 시퀀스당 KV 1.61 GB, GDN fp32 상태 38.6 MB. TP=EP=8 은 플러그인 미검증 | node1 메모리 계획 |
| M4 | eager → CUDA graph 전환은 롤아웃 수치를 바꾼다 | R1 은 eager 로만 통과했다 | 전환 후 R1 재실행 |
| M5 | R3 데이터가 크다 | 토큰당 24층×top-8×int16 = 384 B 로 input_ids 의 48배다. 2,048 샘플×64K 면 약 51 GB, /dev/shm 은 128 GB | 실제 배치로 측정 |
| M6 | 상속된 KL 0.01 이 R3 와 맞지 않는다 | reference logprob 은 의도적으로 replay 하지 않는다 (`megatron_policy_worker.py:1086-1089`). θ=ref 여도 라우팅 차이가 KL 에 들어간다. Ultra 는 KL 0 | D4 |
| M7 | async + in-flight 는 미검증 조건이다 | 진행 중 요청은 이전 가중치의 KV·GDN 상태로 생성을 잇는다. route 에 가중치 버전 표시가 없다. IS 보정은 async 에서 assert 로 강제된다 (`grpo.py:4113`, 상속값 false) | G5 |
| M8 | bias 갱신을 켜면 vLLM 반영 게이트가 없다 | 매 optimizer step 마다 bias 4,608개가 ±1e-3 움직인다. M3 는 bias 불변 상태에서만 통과했다 | D3 와 함께 |
| M9 | RL ckpt → HF → Pai 평가 방향은 미검증이다 | 반출은 tokenizer 를 재직렬화하고 `tokenizer_metadata.json` 을 복사하지 않는다. Gym 경로의 `<\|im_end\|>` 정지는 `generation_config.json` eos [3,0] 병합 하나에 기댄다 | G6 **PASS (수정 뒤, 2026-10-07)** — 변환기 원본은 Pai 가 토크나이저를 못 읽는다(`TokenizersBackend`). 반출은 `tools/export_rl_hf.sh` 로만 한다 (메타데이터는 시작점 복사, 서빙 6/6 `<\|im_end\|>` 정지) |
| M10 | Muon 저장·재개 미검증 | LayerWise 체크포인트는 "fixed DP only" 다 (mcore `megatron/core/optimizer/layer_wise_optimizer.py:935-940`). 같은 CP·DP 로만 재개한다 | G7 **PASS (수정 뒤, 2026-10-07)** — 런 연장 재개가 스케줄러 assert 로 막혀 레시피 `scheduler.max_steps` 고정. 재개 뒤 momentum·Adam 상태·fp32 master 가 이어짐 (초기화 신호 0/62, 음성 대조 50/62) |

### 낮음·운영
- (prompts×gens)/GBS 가 나누어떨어지지 않으면 나머지 샘플이 조용히 빠진다 (`megatron_policy_worker.py:735`).
- Ultra 의 loss 정규화 키 4개(`do_not_average_loss`·`cp_normalize`·`calculate_per_token_loss`·`scale_loss_by_dp_cp_size`)는 NeMo-RL 이 읽지 않는다. 복사해도 효과가 없다.
- `megatron_cfg.moe_router_num_groups`·`moe_router_group_topk` 는 provider 값을 덮는다 (`setup.py:887-892`). null 을 복사하면 그룹 제한이 꺼진다. R3 가 켜져 있으면 KL 에 안 보인다.
- top_p<1 이면 chunked logprob 이 꺼져 128K 메모리 계획이 깨진다. top_p 1.0 고정.
- 잘린 응답도 보상으로 학습한다 (`overlong_filtering: false`). 잘림 비율을 지표로 본다.
- `KNOWN_ISSUES.md` 의 "ES 는 weight transfer 5배 느림" 근거는 FP8 캐시 정리 docstring 한 줄이다 (`megatron_policy_worker.py:2731-2743`). 측정값이 없다.

## 5. 처리량 — recompute·offload 측정 (R5, 2026-10-07 부분 실행 · 이후는 첫 RLVR 런 뒤로 연기)

**사용자 지시 (2026-10-07)**: 속도 최적화는 정확한 RLVR 학습을 먼저 진행한 뒤에 한다. 실제 런에서 학습과 롤아웃 중 무엇이 병목인지 확인하고 적용 여부를 정한다.
아래 수치는 그 판단의 학습측 기준선이다. 합성 데이터(랜덤 토큰)라 실제 데이터에서 다시 확인한다.

조건: 128K/CP8(rank 당 16K 토큰)·EP8, Muon, `expandable_segments`, `logprob_chunk_size 2048` + `fuse_loss`, sub1·main1 8×H100. 정상 상태는 3번째 스텝이다.

| 조건 | 스텝 시간 | 옵티마이저 | 처리량 (노드) | 스텝 후 상주 | peak |
|---|---|---|---|---|---|
| full recompute, 128K 샘플 1개/스텝 | 5.47 s | 0.75 s | 24.0K tok/s | 41.0 GB | 57.2 GB |
| full recompute, 128K 샘플 4개/스텝 | 15.7 s | 0.71 s | **33.4K tok/s** | 41.0 GB | 57.2 GB |
| 위 + R3 (합성 group-limited route, main1) | 16.5 s | 1.52 s | 31.7K tok/s | 41.0 GB | 57.3 GB |
| 64K/CP8 full recompute, 64K 샘플 8개/스텝 | 12.2 s | 0.78 s | 43.1K tok/s | 38.9 GB | 50.8 GB |
| selective `layernorm,moe` · `moe` · `layernorm` · recompute 없음 (128K, R3 유무 모두) | — | — | — | — | **OOM** |
| selective `layernorm,moe` · `moe` · recompute 없음 (64K) | — | — | — | — | **OOM** |

- 첫 보고의 "약 15K tok/s"는 R4 가 2번째 스텝(워밍업)과 샘플 1개 스텝을 잰 값이다. 정상 상태 full recompute 는 33.4K tok/s 다.
- R4 의 "고정 바닥 27 GB"는 Muon momentum 지연 할당 전 값이다. 스텝 후 상주는 41 GB 다.
- 128K 에서는 attention 이 FLOPs 의 약 2/3 다. 64K(43.1K) 보다 128K(33.4K) 가 느린 이유다.
- selective OOM 지점: 랜덤 토큰 + live router 에서는 MoE 토큰 정렬 버퍼(1.56 GiB = 409K 토큰, 평균의 3.1배, 매번 같은 EP 랭크)였다.
  균형 route(R3)에서는 RL loss 의 logprob 계산·backward 로 옮겨 갔다. 랜덤 토큰 쏠림은 측정 아티팩트이고, 주원인은 selective 저장 activation + RL loss 메모리다.
- Pai SFT 는 64K/CP8 selective 를 offload 없이 52.2 GB 로 통과했다 (Pai `docs/gdn_cp_port.md`). NeMo-RL 은 같은 조건에서 OOM 이다.
  차이 후보는 RL loss 의 logprob backward(SFT 는 TE fused CE)와 저장 activation 구성이다. optimizer offload(정상 상태 약 16 GB 추정)만으로는 부족하다.
- 속도 상한: recompute 를 없애도 이득은 계산상 약 25% 다 (Pai 실측 +15%). Pai 와의 나머지 차이는 프로파일로 찾는다.
- 연기한 항목 (병목이 학습으로 확인되면): 메모리 스냅샷 분해 → RL loss 메모리 절감 → optimizer offload 포팅(Pai 기능 #4) → 한 스텝 프로파일.
  병목이 롤아웃이면 vLLM 쪽(CUDA graph·배치 파라미터·prefix caching·DP 배치)을 먼저 본다.
- 산출물: `$NRL_ROOT/gates/train_throughput/` (`run_case.sh`, `matrix_*.sh`, 조건별 `.json`·`.log`).

### 5.1 첫 RLVR 런 병목 판정 (2026-10-08 00:05, 구간 1 의 1~4스텝)

| 스텝 | 스텝 시간 | 학습 | logprob | 롤아웃 대기 (`exposed_generation`) | 동기화 | vLLM 최소 유휴 | 배치 토큰 |
|---|---|---|---|---|---|---|---|
| 1 | 1,291 s | 750 | 411 | 79 | 28 | 1,258 s | 23.3M |
| 2 | 1,267 s | 800 | 332 | 83 | 29 | 1 s | 28.4M |
| 3 | 1,786 s | 698 | 311 | 725 | 28 | 641 s | 23.3M |
| 4 | 867 s | 515 | 262 | 43 | 29 | 2 s | 15.6M |
| (2~4 평균) | 1,307 s | 671 (51%) | 301 (23%) | 284 (22%) | 28 (2%) | | |
| 5 | 2,901 s | 754 | 327 | 1,767 | 28 | 746 s | — |
| **3~5 평균 (정상 상태)** | **1,851 s** | 655 (35%) | 300 (16%) | **845 (46%)** | 28 (2%) | | |

판정 (5스텝 뒤 수정, 2026-10-08 01:05): 1~4스텝으로는 학습 쪽(74%)이 주 병목으로 보였지만 5스텝에서 롤아웃 대기가 1,767 s(61%)였다. 정상 상태(3~5스텝)는 **롤아웃 긴 꼬리 46% · 학습 + logprob 51%** 로 둘이 비슷하고, 대기가 격스텝으로 커진다. 매 배치에 128K 까지 가는 샘플이 있어(최대 생성 130.9K, 1~5스텝 모두) 배치 완성이 그 샘플의 순차 디코딩(45~60분)에 묶인다. 첫 판정(1~4스텝)은 다음과 같았다 — 학습 쪽이 주 병목 (학습 + logprob 74%). 롤아웃 대기는 격스텝으로 생긴다 — `max_trajectory_age_steps 1` 이라 다음 배치를 가중치 갱신 뒤에야 시작하고,
배치 완료를 가장 긴 응답이 정한다. "vLLM 최소 유휴"(생성 GPU 8장 모두가 요청 없이 쉰 시간)가 1·3스텝에 10~20분이라 생성 능력은 남는다.
학습 처리량은 노드당 3.0만~3.6만 tok/s (R5 의 128K full recompute 3.34만과 같은 수준)이고 시간은 배치 토큰에 비례한다. 구간 1 은 덤프 기록에 스텝당 ≈ 2분이 더 든다.
정상 상태 속도(스텝 ≈ 31분)면 1에폭(≈ 1,120스텝)은 ≈ 24일이다. 다음: 롤아웃 꼬리(최대 생성 길이 D7·`max_trajectory_age_steps` 2)와 학습 가속(prev_logprob 패스·R3 route 전송)을 함께 검토한다.

## 6. 게이트 계획 (2026-10-07 재편 — 정확성 → 첫 RLVR 런 → 속도)

**1단계 — 정확성 (첫 RLVR 런 전 필수)**

| # | 게이트 | 노드 | 상태 |
|---|---|---|---|
| G0 | `student_rlvr1_alpha.yaml` 골격 · `tools/check_alpha_recipe.py` · R4 하네스 수정 | CPU | **완료** — 골격 검사 ERROR 0 / WARN 2, Ultra 함정 5개 주입 시 전부 ERROR |
| G1 | packing 상태 누출 판별 (`tools/verify_packing_isolation.py`) | 1노드 | **PASS** — fla·FlashQLA 모두 4개 길이 쌍에서 B logprob 비트 동일 |
| G2 | packing+CP 경로 정합: CP8+packing 으로 R1 (KL < 0.002, 위치 구간 평탄) + R3 trace 검증 (`NRL_R3_TRACE`·`tools/check_r3_trace.py`). 긴 생성 길이에서 위치별 KL | 1노드 | **PASS** — 4K CP8+packing 0.0015/0.0013/0.0014, R3 forward 검증 불일치 0 · 32K 생성 0.0016, 위치 16–32K 0.00166 (평탄) |
| G3 | Gym 경로: 서버 파서 스모크(도구 호출 파싱·추론 분리·invalid 판정률) → `strict` 유지 구현(결정 13) → R3 P6 실경로 | 1노드 | **PASS (수정 뒤)** — KL 0.0017/0.0016, invalid·malformed 0/64, strict 렌더 36/36. 결함 5건 수정 — 그중 vLLM strict 도구 제약 디코딩(H3 의 새 형태)이 가장 중대 |
| G5 | 실레시피 2노드 스모크: async + in-flight + Gym + 분리 토폴로지. refit 시간·KL·R3 누락 0·타이밍 지표 | 2노드 | **PASS (수정 뒤)** — KL 0.0018/0.0018/0.0019, refit 28.4 s, route 누락 0. 비동기 저장 불가(ES) → 동기 저장 |
| G6 | G5 체크포인트로 HF 반출 → Pai forward_sanity·서빙 | 1노드 | **PASS (수정 뒤)** — 변환기 원본은 Pai 가 토크나이저를 못 읽는다 → `tools/export_rl_hf.sh` (메타데이터 시작점 복사). 가중치 오류 0·동결 텐서 비트 동일, forward 차이는 잡음 바닥, 서빙 6/6 |
| G7 | Muon 저장 → 재개 (런 연장: max_num_steps 3→5) · 옵티마이저 상태 연속성 (`tools/verify_optimizer_resume.py`) | 2노드 | **PASS (수정 뒤)** — 첫 시도는 재개에서 스케줄러 assert → `scheduler.max_steps` 고정. step·데이터·replay buffer·lr 이어짐, KL 0.0019/0.0018, 옵티마이저 상태 초기화 신호 0/62 |

**2단계 — 첫 RLVR 런 (정확성 확인 + 병목 측정)**
- 결정 D1~D7 반영 레시피로 실행한다. 정확성 지표: rollout↔train KL(`seq_logprob_error`·위치별), R3 경고 0, 보상·잘림 비율, invalid tool call·malformed think 비율.
- 병목 지표 (async GRPO 타이밍): `timing/train/exposed_generation`(학습이 롤아웃을 기다린 시간) · `policy_training` · `policy_and_reference_logprobs` · `weight_sync`, vLLM 지표(진행 중 배치·대기 샘플), 두 노드 GPU 사용률.
  `exposed_generation` 이 크면 롤아웃 병목, 0 에 가깝고 버퍼가 차 있으면 학습 병목이다.
- async 겹침 해석 (`nemo_rl/algorithms/async_utils/trajectory_collector.py`, 2026-10-07 확인): 시작할 때 1·2스텝용 배치를 함께 생성한다 (`max_trajectory_age_steps 1`).
  그 뒤에는 스텝 N 을 학습하는 동안 직전 가중치로 다음 배치를 만든다. 동시에 진행되는 배치는 최대 2개이고, 가중치 동기화는 진행 중인 생성을 기다리지 않는다 (in-flight).
  GRPO 는 그룹 16개 응답이 다 끝나야 학습하므로 배치 완료는 가장 긴 응답이 정한다. 1스텝은 겹칠 대상이 없는 첫 배치 대기라 병목 판정에서 뺀다.
  생성이 학습보다 길면 스텝 간격 ≈ (배치 생성 + 학습 + 동기화) / 2 이고, 짧으면 학습 시간이 간격을 정한다 (수집기 `idle/generation_limit_pause` 증가).
  `exposed_generation` 은 학습 루프가 replay buffer 에서 배치를 기다린 시간이다.
- 초기 체크포인트를 Pai 벤치로 평가해 비하락을 확인한다.

**2단계 운영 (2026-10-07 21:19 시작, 커밋 `546b2b549`)**: 실행기는 `$NRL_ROOT/runs/rlvr1_alpha/` 에 있다.
`run_rlvr1.sh <tag> [override]` 는 `gates/two_node/run_g5.sh` 와 같은 2노드 환경으로 기동하고, 기동 전에 GPU 점유를 검사한다(1 GiB).
`run_first.sh` 는 구간 1 과 구간 2 를 잇는다. 모든 구간이 같은 `ckpt/` 를 쓰므로 다시 실행하면 최신 step 에서 재개한다 (G7).
- 구간 1 `seg1`: `grpo.max_num_steps=10 env.should_log_nemo_gym_responses=false` — 학습 데이터 덤프(`seg1/exp_*/train_data_step*.jsonl`)와 R3 trace 를 남긴다.
- 구간 2 `seg2`: 덤프·trace 를 끄고 레시피 기본값(1에폭 ≈ 1,120 스텝)까지 이어 간다. route 검증(`NRL_ROUTER_REPLAY_VALIDATE=1`)은 계속 켠다.
- HF 반출: `tools/export_watch.sh ckpt hf 100 300` 이 100스텝 체크포인트를 `hf/hf_step_NNNNN` 으로 반출한다 (CPU, 1회 30 GB, 모두 보존).
  체크포인트는 2번의 저장(20스텝) 동안 남으므로 반출(≈ 4~6분)이 끝나기 전에 지워지지 않는다.
- 체크포인트 관리: NeMo-RL 은 새 체크포인트를 `tmp_step_N` 에 다 쓰고 이름을 바꾼 뒤 오래된 것을 지운다 (`finalize_checkpoint` → `remove_old_checkpoints`).
  그래서 저장하는 동안에는 보존 개수 + 1 개가 디스크에 있다. 보존은 최근 2개다 (사용자 결정 2026-10-07). 용량 예산은 `STATUS.md` "첫 RLVR 런 저장 용량" 표에 둔다.

**3단계 — 속도 최적화 (병목 쪽만)**: §5 의 연기 항목. 학습 병목이면 R5 후속, 롤아웃 병목이면 vLLM 레버 (M2·M4·D2 포함, 바꿀 때마다 R1).

## 7. 결정 (2026-10-07 사용자 결정 — `RL_PLAN.md` 결정 15)

| # | 결정 | 결과 |
|---|---|---|
| D1 | judge·sandbox 의존 행 23.4% + nvarc 4.2% | **첫 런은 judge 불필요 10개 환경만** — `rlvr1_alpha_judgefree.jsonl` 71,730행 (D1 구조 게이트 OK). 도구 호출 환경이 52.9% 가 된다. identity·GenRM·judge 환경은 judge 배치 결정 뒤 |
| D2 | prefix caching | 권고값(끔) 유지 — upstream R3 레시피와 같음 |
| D3 | 부하 균형 | **expert bias 갱신 0** (R1·G2 검증 조건) |
| D4 | KL | **0 + `seq_logprob_error_threshold: 2`** (rollout↔train 정합 지표 유지) |
| D5 | lr·warmup | **Muon lr 1e-6, warmup 10 GRPO step** |
| D6 | ES 기본값 | 권고값(정책 워커에 켬) 유지 — G5 에서 refit 시간 확인 |
| D7 | 길이 | **128K 바로** |
